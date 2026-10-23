#!/usr/bin/env python3
"""
Feature extraction pipeline for CNC controller data.

For every dataset, controller signals (feedrate, spindle load, axis loads, ...)
are accumulated cut by cut, and at each checkpoint a set of statistical and
dynamic features is computed over the cumulative window. Two window-level
features are also computed:

    N_STEPS       Total number of steps in the window (sum of the last
                  ``step_no`` of each cut, robust to missing rows).
    DURATION_SEC  Elapsed time from the first ``start_step`` of the first cut
                  to the last ``end_step`` of the last cut in the window.

Datasets whose first cut is missing are corrected by adding the contribution of
a typical first cut to the features that accumulate over cuts. The counts
(``DCOUNT_*`` and ``N_STEPS``) are increased by their median value at cut 1.
``DURATION_SEC`` is increased by the median time from the start of cut 1 to
the start of cut 2, i.e. the duration of the first cut plus the pause that
follows it. These values are always estimated from the training set, so no
information leaks from evaluation data.

Numerics: signals are processed in float32, per-variable results are stored in
a long table that is serialised as CSV before pivoting, and the wide table has
its columns sorted alphabetically (feature, then variable). These steps are
kept on purpose so that the output is identical to the reference pipeline.

Example
-------
    python extract_controller_features.py \\
        --train-controller-dir Training_Dataset/Controller_Data \\
        --eval-controller-dir Evaluation_Dataset/Controller_Data \\
        --output-dir features/
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from data_loader_evalset import DataLoader as EvalDataLoader
from data_loader_trainset import DataLoader as TrainDataLoader

logger = logging.getLogger(__name__)

CONTROLLER_COLS = (
    "feedrate", "mainSpndLoad", "mainSpndSpd", "X_load", "Y_load", "Z_load", "B_load",
)
FEATURE_NAMES = (
    "XMAX", "XMIN", "XMEAN", "XMEDIAN", "XRMS", "XV", "XSTD", "XSKEW", "XKURT",
    "XP2P", "XMAD", "XIQR", "XENTROPY", "XCREST", "XSHAPE", "XCLEAR", "XENERGY",
    "DMEAN", "DSTD", "DRMS", "DMAX", "DCOUNT", "XPEAKS", "XZC",
)
KEY_COLS = ["dataset", "cut"]
N_CUTS = 26
CHECKPOINTS = frozenset({1, 6, 11, 16, 21, 26})
EPS = 1e-12

TRAIN_OUTPUT_NAME = "features_controller_train.csv"
EVAL_OUTPUT_NAME = "features_controller_eval.csv"


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #

def signal_features(signal: np.ndarray) -> list:
    """Statistical and dynamic features of one controller signal.

    The signal is used in its own dtype (float32) and NaNs are not removed.
    """
    signal = np.asarray(signal)
    if signal.size < 2 or np.all(np.isnan(signal)):
        return [np.nan] * len(FEATURE_NAMES)

    # Basic statistics
    xmax = np.max(signal)
    xmin = np.min(signal)
    xmean = np.mean(signal)
    xmedian = np.median(signal)
    xrms = np.sqrt(np.mean(signal ** 2))
    xv = np.var(signal, ddof=1)
    xstd = np.std(signal, ddof=1)
    xskew = skew(signal)
    xkurt = kurtosis(signal)
    xp2p = np.ptp(signal)
    xmad = np.mean(np.abs(signal - xmean))
    xiqr = np.percentile(signal, 75) - np.percentile(signal, 25)

    # Dynamics
    diff = np.diff(signal)
    dmean = np.mean(diff)
    dstd = np.std(diff)
    drms = np.sqrt(np.mean(diff ** 2))
    dmax = np.max(np.abs(diff))
    dcount = np.sum(np.abs(diff) > (0.1 * np.ptp(signal)))

    # Shape
    abs_signal = np.abs(signal)
    abs_sum = np.sum(abs_signal) + EPS
    xentropy = -np.sum((abs_signal / abs_sum) * np.log(abs_signal / abs_sum + EPS))
    xcrest = xmax / (xrms + EPS)
    xshape = xrms / (np.mean(abs_signal) + EPS)
    xclear = xmax / (np.mean(np.sqrt(abs_signal)) + EPS)
    xenergy = np.sum(signal ** 2)
    xpeaks = len(find_peaks(signal)[0])
    xzc = ((signal[:-1] * signal[1:]) < 0).sum()

    return [
        xmax, xmin, xmean, xmedian, xrms, xv, xstd, xskew, xkurt, xp2p,
        xmad, xiqr, xentropy, xcrest, xshape, xclear, xenergy,
        dmean, dstd, drms, dmax, dcount, xpeaks, xzc,
    ]


def window_variable_features(window: pd.DataFrame, dataset: int, cut: int) -> pd.DataFrame:
    """Long table: one row per controller variable with all its features."""
    rows = [{"variable": var, **dict(zip(FEATURE_NAMES, signal_features(window[var].values)))}
            for var in CONTROLLER_COLS]
    return pd.DataFrame(rows).assign(dataset=dataset, cut=cut)


def window_extra_features(cuts: list[pd.DataFrame]) -> dict:
    """``N_STEPS`` and ``DURATION_SEC`` of a cumulative window."""
    n_steps = sum(int(df["step_no"].iloc[-1]) for df in cuts if "step_no" in df.columns)
    start = pd.to_datetime(cuts[0]["start_step"].iloc[0]) if "start_step" in cuts[0] else pd.NaT
    end = pd.to_datetime(cuts[-1]["end_step"].iloc[-1]) if "end_step" in cuts[-1] else pd.NaT
    duration = (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else np.nan
    return {"N_STEPS": n_steps, "DURATION_SEC": duration}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def extract_features(loader, dataset_ids: Iterable[int]) -> pd.DataFrame:
    """Cumulative-window features for every (dataset, checkpoint), in wide format."""
    long_parts, extra_rows = [], []
    for dataset in dataset_ids:
        cuts: list[pd.DataFrame] = []
        for cut in range(1, N_CUTS + 1):
            logger.debug("Dataset %d, cut %d", dataset, cut)
            try:
                cut_df = loader.get_controller_data(dataset, cut)
            except Exception as exc:  # loader-specific errors
                logger.warning("No controller data for dataset %d, cut %d (%s); skipping.",
                               dataset, cut, exc)
                cut_df = None

            if cut_df is not None and not cut_df.empty:
                num_cols = cut_df.select_dtypes(include=[np.number]).columns
                cut_df[num_cols] = cut_df[num_cols].astype(np.float32)
                cuts.append(cut_df)

            if cut in CHECKPOINTS:
                if not cuts:
                    logger.warning("Dataset %d, checkpoint %d has no accumulated data; "
                                   "row omitted.", dataset, cut)
                    continue
                window = pd.concat(cuts, ignore_index=True)
                long_parts.append(window_variable_features(window, dataset, cut))
                extra_rows.append({**window_extra_features(cuts), "dataset": dataset, "cut": cut})

    if not long_parts:
        raise RuntimeError("No features were extracted: every checkpoint was empty.")

    # The long table goes through CSV text before pivoting, as in the reference
    # pipeline; this fixes how the float32 results are rounded.
    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = pd.read_csv(io.StringIO(long_df.to_csv(index=False)))

    wide = long_df.pivot_table(index=KEY_COLS, columns="variable", values=list(FEATURE_NAMES))
    wide.columns = [f"{feature}_{var}" for feature, var in wide.columns]
    return wide.reset_index().merge(pd.DataFrame(extra_rows), on=KEY_COLS, how="left")


def median_first_cut_span(loader, dataset_ids: Iterable[int]) -> float | None:
    """Median time in seconds from the start of cut 1 to the start of cut 2.

    This is what ``DURATION_SEC`` misses when the first cut is unavailable: the
    duration of cut 1 plus the pause before cut 2. Returns ``None`` if no
    dataset has both cuts.
    """
    spans = []
    for dataset in dataset_ids:
        try:
            starts = [pd.to_datetime(loader.get_controller_data(dataset, cut)["start_step"].iloc[0])
                      for cut in (1, 2)]
        except Exception:  # missing cut, column or rows: the dataset is not used
            continue
        span = (starts[1] - starts[0]).total_seconds()
        if pd.notna(span):
            spans.append(span)
    return float(np.median(spans)) if spans else None


def correct_missing_first_cut(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    count_cols: list[str],
    offsets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add a typical cut-1 contribution to datasets lacking cut 1.

    ``count_cols`` are increased by their median over the cut-1 rows of
    ``reference``; the columns in ``offsets`` are increased by the given values.
    """
    missing = set(df["dataset"]) - set(df.loc[df["cut"] == 1, "dataset"])
    if not missing:
        return df

    ref_cut1 = reference.loc[reference["cut"] == 1, list(count_cols)]
    if ref_cut1.empty:
        raise ValueError("Reference data has no cut=1 rows; cannot estimate the contribution.")

    contribution = pd.concat([ref_cut1.median(), pd.Series(offsets or {}, dtype=float)])
    cols = list(contribution.index)
    logger.info("Datasets without cut=1: %s. Adding the cut-1 contribution estimated from "
                "%d reference datasets to %d columns.", sorted(missing), len(ref_cut1), len(cols))

    df = df.copy()
    df[cols] = df[cols].astype(float)  # integer counts may receive a fractional median
    mask = df["dataset"].isin(missing)
    df.loc[mask, cols] = df.loc[mask, cols] + contribution.to_numpy()
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cumulative-window features from CNC controller data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-controller-dir", type=Path, required=True,
                        help="Directory with the training controller data.")
    parser.add_argument("--eval-controller-dir", type=Path, required=True,
                        help="Directory with the evaluation controller data.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where the feature CSV files are written.")
    parser.add_argument("--train-sensor-dir", type=Path, default=None,
                        help="Training sensor data directory "
                             "(default: <train-controller-dir>/../Sensor_Data).")
    parser.add_argument("--eval-sensor-dir", type=Path, default=None,
                        help="Evaluation sensor data directory "
                             "(default: <eval-controller-dir>/../Sensor_Data).")
    parser.add_argument("--train-datasets", type=int, nargs="+", default=list(range(1, 7)),
                        help="Training dataset IDs to process.")
    parser.add_argument("--eval-datasets", type=int, nargs="+", default=list(range(1, 4)),
                        help="Evaluation dataset IDs to process.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable per-cut debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    train_sensor_dir = args.train_sensor_dir or args.train_controller_dir.parent / "Sensor_Data"
    eval_sensor_dir = args.eval_sensor_dir or args.eval_controller_dir.parent / "Sensor_Data"

    splits = {
        "train": (TrainDataLoader(str(args.train_controller_dir), str(train_sensor_dir),
            set_available=list(args.train_datasets)), args.train_datasets, TRAIN_OUTPUT_NAME),
        "eval": (EvalDataLoader(str(args.eval_controller_dir), str(eval_sensor_dir),
            set_available=list(args.eval_datasets)), args.eval_datasets, EVAL_OUTPUT_NAME),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (loader, dataset_ids, filename) in splits.items():
        start = time.perf_counter()
        paths[name] = args.output_dir / filename
        extract_features(loader, dataset_ids).to_csv(paths[name], index=False)
        logger.info("[%s] Features extracted in %.2f s.", name, time.perf_counter() - start)

    # Correction reads the saved files back; the training set is always the reference.
    features = {name: pd.read_csv(path) for name, path in paths.items()}
    reference = features["train"]
    count_cols = [c for c in reference.columns if c.startswith("DCOUNT_")] + ["N_STEPS"]

    train_loader, train_ids, _ = splits["train"]
    cut1_duration = float(reference.loc[reference["cut"] == 1, "DURATION_SEC"].median())
    span = median_first_cut_span(train_loader, train_ids)
    if span is None:
        logger.warning("Could not measure the time between cuts 1 and 2 in the training set; "
                       "using the cut-1 duration only.")
        span = cut1_duration
    logger.info("Training set, medians: cut-1 duration %.1f s; start of cut 1 to start of "
                "cut 2 %.1f s (pause of about %.1f s).", cut1_duration, span, span - cut1_duration)

    for name in ("eval", "train"):
        corrected = correct_missing_first_cut(features[name], reference, count_cols,
                                              offsets={"DURATION_SEC": span})
        corrected.to_csv(paths[name], index=False)
        logger.info("[%s] Saved %d rows to %s", name, len(corrected), paths[name])


if __name__ == "__main__":
    main()
