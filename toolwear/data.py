"""Loading and shaping of sensor features, controller features and wear labels."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ID_COLS, PREDICT_CHECKPOINTS

logger = logging.getLogger(__name__)

KEY_COLS = ["dataset", "cut"]

Sequences = dict[int, np.ndarray]
Metas = dict[int, list[tuple[int, int]]]


# --------------------------------------------------------------------------- #
# Sensor features
# --------------------------------------------------------------------------- #

def build_sequences(
    df: pd.DataFrame, feature_cols: Sequence[str] | None = None
) -> tuple[Sequences, Metas, list[str]]:
    """Turn windowed sensor features into one time-ordered sequence per dataset.

    Args:
        df: Output of the sensor feature extraction (one row per window).
        feature_cols: Columns to use. Defaults to every non-ID column; pass the
            training columns when building sequences for new data.

    Returns:
        ``(sequences, metas, feature_cols)`` where ``sequences[d]`` has shape
        ``(n_windows, n_features)`` and ``metas[d][i]`` is the ``(cut, window)``
        of row ``i``. Windows with every feature missing are dropped.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ID_COLS]
    feature_cols = list(feature_cols)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Sensor features are missing {len(missing)} columns, e.g. {sorted(missing)[:5]}")

    sequences: Sequences = {}
    metas: Metas = {}
    for dataset_id, group in df.groupby("dataset"):
        group = group.sort_values(["cut", "window"])
        group = group[~group[feature_cols].isna().all(axis=1)]
        sequences[int(dataset_id)] = group[feature_cols].to_numpy(dtype=np.float64)
        metas[int(dataset_id)] = list(zip(group["cut"].astype(int), group["window"].astype(int)))
    return sequences, metas, feature_cols


def label_position(meta: Sequence[tuple[int, int]], cut: int) -> tuple[int | None, bool]:
    """Index of the last window of ``cut`` in a sequence (causal position).

    Falls back to the last window of an earlier cut when ``cut`` has no data.
    Returns ``(position, exact_match)``; position is ``None`` if nothing precedes it.
    """
    exact = [i for i, (c, _) in enumerate(meta) if c == cut]
    if exact:
        return exact[-1], True
    prior = [i for i, (c, _) in enumerate(meta) if c <= cut]
    return (prior[-1], False) if prior else (None, False)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def parse_dataset_id(set_no: object) -> int:
    match = re.search(r"(\d+)\s*$", str(set_no))
    if match is None:
        raise ValueError(f"Cannot parse a dataset id from {set_no!r}")
    return int(match.group(1))


def load_labels(path: str) -> pd.DataFrame:
    """Load wear measurements as ``[dataset, cut, measurement]``."""
    df = pd.read_csv(path).dropna(subset=["Measurement"])
    return pd.DataFrame({
        "dataset": df["Set_No"].map(parse_dataset_id).astype(int),
        "cut": df["Cut_No"].astype(int),
        "measurement": df["Measurement"].astype(float),
    }).reset_index(drop=True)


def attach_positions(labels: pd.DataFrame, metas: Metas) -> pd.DataFrame:
    """Add the sequence ``position`` of each label; drop labels with no data."""
    labels = labels.copy()
    positions = []
    for ds, cut in zip(labels["dataset"], labels["cut"]):
        pos, exact = label_position(metas.get(int(ds), []), int(cut))
        if pos is None:
            logger.warning("No sensor data for dataset %d, cut %d; label dropped.", ds, cut)
        elif not exact:
            logger.warning("Dataset %d, cut %d: using the last valid earlier window.", ds, cut)
        positions.append(pos)
    labels["position"] = positions
    return labels.dropna(subset=["position"]).astype({"position": int}).reset_index(drop=True)


def build_transitions(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per pair of consecutive measured cuts.

    Columns: ``dataset, cut, prev_cut, delta_measurement, measurement_cur,
    position, prev_delta, prev_cumulative``. ``prev_*`` are the true values of
    the autoregressive inputs (teacher forcing).
    """
    rows = []
    for ds, group in labels.groupby("dataset"):
        group = group.sort_values("cut")
        cuts = group["cut"].tolist()
        meas = dict(zip(group["cut"], group["measurement"]))
        pos = dict(zip(group["cut"], group["position"]))
        prev_delta = 0.0
        for prev_cut, cut in zip(cuts[:-1], cuts[1:]):
            delta = meas[cut] - meas[prev_cut]
            rows.append({
                "dataset": int(ds), "cut": cut, "prev_cut": prev_cut,
                "delta_measurement": delta, "measurement_cur": meas[cut],
                "position": pos[cut], "prev_delta": prev_delta,
                "prev_cumulative": meas[prev_cut],
            })
            prev_delta = delta
    return pd.DataFrame(rows)


def initial_measurements(labels: pd.DataFrame) -> dict[int, float]:
    """Measured wear at cut 1 per dataset (the anchor of every prediction)."""
    cut1 = labels[labels["cut"] == 1]
    return {int(d): float(m) for d, m in zip(cut1["dataset"], cut1["measurement"])}


def build_prediction_grid(
    metas: Metas, checkpoints: Iterable[int] = PREDICT_CHECKPOINTS
) -> pd.DataFrame:
    """``[dataset, cut, position]`` rows for every checkpoint that has sensor data."""
    rows = []
    for ds in sorted(metas):
        for cut in checkpoints:
            pos, _ = label_position(metas[ds], cut)
            if pos is None:
                logger.warning("No sensor data for dataset %d, cut %d; cannot predict there.", ds, cut)
                continue
            rows.append({"dataset": ds, "cut": cut, "position": pos})
    return pd.DataFrame(rows, columns=["dataset", "cut", "position"])


def anchor_matrix(
    frame: pd.DataFrame, anchors: Mapping[int, float], fallback: float
) -> pd.DataFrame:
    """``cut_number`` and ``initial_measurement`` features for each row of ``frame``."""
    initial = frame["dataset"].map(anchors).astype(float)
    if initial.isna().any():
        missing = sorted(frame.loc[initial.isna(), "dataset"].unique().tolist())
        logger.warning("No cut-1 measurement for datasets %s; using the training median.", missing)
        initial = initial.fillna(fallback)
    return pd.DataFrame({
        "cut_number": frame["cut"].astype(float).to_numpy(),
        "initial_measurement": initial.to_numpy(),
    })


# --------------------------------------------------------------------------- #
# Controller features
# --------------------------------------------------------------------------- #

def controller_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLS]


def prune_correlated_features(
    df: pd.DataFrame, feature_cols: Sequence[str], threshold: float
) -> list[str]:
    """Drop features whose absolute correlation with an earlier one exceeds ``threshold``."""
    corr = df[list(feature_cols)].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped = {col for col in upper.columns if (upper[col] > threshold).any()}
    kept = [c for c in feature_cols if c not in dropped]
    logger.info("Controller features: %d -> %d after correlation pruning (|r| > %.2f).",
                len(feature_cols), len(kept), threshold)
    return kept


def controller_matrix(
    frame: pd.DataFrame, controller_df: pd.DataFrame, feature_cols: Sequence[str]
) -> pd.DataFrame:
    """Controller features aligned with the ``(dataset, cut)`` rows of ``frame``."""
    feature_cols = list(feature_cols)
    merged = frame[KEY_COLS].merge(controller_df[KEY_COLS + feature_cols], on=KEY_COLS, how="left")
    missing = merged[feature_cols].isna().all(axis=1)
    if missing.any():
        keys = list(merged.loc[missing, KEY_COLS].itertuples(index=False, name=None))
        logger.warning("No controller features for %s; they will be median-imputed.", keys)
    return merged[feature_cols].reset_index(drop=True)