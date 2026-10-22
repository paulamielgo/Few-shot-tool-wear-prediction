"""Automatic termination of hyperparameter optimisation.

Implements the criterion of Makarova et al., "Automatic Termination for
Hyperparameter Optimization" (AutoML 2022), with Optuna's own evaluators:

* Regret bound: :class:`optuna.terminator.RegretBoundEvaluator` (unchanged). A
  GP fitted on the top trials bounds how much the objective could still improve.
* Statistical error of the objective: the searched objective is the robust CV
  score ``mean + std`` over folds, so its error includes both terms::

      var_mean = (1/k + 1/(k-1)) * s²     Nadeau & Bengio (2003), as in Optuna's
                                          CrossValidationErrorEvaluator
      var_std  = s² / (2 (k - 1))         sampling variance of the fold std
      error    = sqrt(var_mean + var_std)

  where ``s²`` is the variance of the best trial's fold scores.

The search stops once ``regret_bound < error``. Optuna's standard
:class:`~optuna.terminator.CrossValidationErrorEvaluator` (mean only) is also
computed and logged as a diagnostic; it never decides the stop.

Requires Optuna >= 4.0.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import optuna
from optuna.terminator import CrossValidationErrorEvaluator, RegretBoundEvaluator

from .config import BASE_SEED, TerminationConfig

logger = logging.getLogger(__name__)

HISTORY_ATTR = "termination_history"
STOPPED_ATTR = "termination_stopped"


def robust_cv_error(fold_scores: Sequence[float]) -> float:
    """Standard deviation of the robust CV score ``mean + std`` of k folds."""
    scores = np.asarray(fold_scores, dtype=np.float64)
    k = len(scores)
    if k < 2:
        raise ValueError("At least 2 fold scores are needed to estimate the CV error.")
    var = float(np.var(scores, ddof=0))
    var_mean = (1.0 / k + 1.0 / (k - 1)) * var
    var_std = var / (2.0 * (k - 1))
    return math.sqrt(var_mean + var_std)


class RobustObjectiveErrorEvaluator:
    """Error of the best trial's robust CV score, read from ``cv_fold_scores``.

    Same interface as the evaluators in :mod:`optuna.terminator`.
    """

    def evaluate(self, trials: list[optuna.trial.FrozenTrial],
                 study_direction: optuna.study.StudyDirection) -> float:
        complete = [t for t in trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not complete:
            raise ValueError("No completed trials.")
        pick = max if study_direction == optuna.study.StudyDirection.MAXIMIZE else min
        best = pick(complete, key=lambda t: t.value)
        scores = best.user_attrs.get("cv_fold_scores")
        if not scores:
            raise ValueError(f"Best trial {best.number} has no 'cv_fold_scores'.")
        return robust_cv_error(scores)


class RegretBoundStopper:
    """Optuna callback that stops the study with the Makarova et al. criterion.

    Trials must store their fold scores in the user attribute ``cv_fold_scores``
    and report them with :func:`optuna.terminator.report_cross_validation_scores`
    (used by the diagnostic standard evaluator). Every check is stored in the
    study user attribute ``termination_history``, so it survives resumptions.
    """

    def __init__(self, config: TerminationConfig, seed: int = BASE_SEED) -> None:
        self.config = config
        self.improvement_evaluator = RegretBoundEvaluator(
            top_trials_ratio=config.top_trials_ratio, min_n_trials=config.min_trials, seed=seed)
        self.error_evaluator = RobustObjectiveErrorEvaluator()
        self.standard_error_evaluator = CrossValidationErrorEvaluator()

    @staticmethod
    def already_stopped(study: optuna.Study) -> bool:
        return bool(study.user_attrs.get(STOPPED_ATTR, False))

    @staticmethod
    def history(study: optuna.Study) -> list[dict]:
        return list(study.user_attrs.get(HISTORY_ATTR, []))

    def check(self, study: optuna.Study) -> dict | None:
        """Evaluate the criterion; ``None`` before ``min_trials`` or if it cannot be computed."""
        complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if len(complete) < self.config.min_trials:
            return None

        try:
            regret = float(self.improvement_evaluator.evaluate(
                trials=complete, study_direction=study.direction))
            if self.config.threshold is not None:
                error = self.config.threshold
            else:
                error = float(self.error_evaluator.evaluate(
                    trials=complete, study_direction=study.direction))
        except Exception as exc:  # the search must go on if the check fails
            logger.warning("Termination check failed after %d trials: %s", len(complete), exc)
            return None

        try:
            standard = float(self.standard_error_evaluator.evaluate(
                trials=complete, study_direction=study.direction))
        except Exception:
            standard = None

        return {
            "trial_number": max(t.number for t in complete),
            "n_complete": len(complete),
            "best_value": study.best_value,
            "regret_bound": regret,
            "threshold": error,
            "cv_error_standard": standard,
            "stop": regret < error,
        }

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        status = self.check(study)
        if status is None:
            return

        study.set_user_attr(HISTORY_ATTR, self.history(study) + [status])
        logger.info("Termination check after %d trials: regret bound=%.4g, threshold=%.4g",
                    status["n_complete"], status["regret_bound"], status["threshold"])
        if status["stop"]:
            study.set_user_attr(STOPPED_ATTR, True)
            logger.info("Stopping criterion met after %d trials: the remaining regret is "
                        "below the CV estimation error.", status["n_complete"])
            study.stop()