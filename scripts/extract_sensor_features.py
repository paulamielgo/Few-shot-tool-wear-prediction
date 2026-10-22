#!/usr/bin/env python3
"""
Feature extraction pipeline for CNC sensor data.

Each cut is split into K equal-duration time windows. For every window and
sensor channel (accelerations X/Y/Z and acoustic emission) the following
feature blocks are computed:

    Temporal   Statistical descriptors of the raw signal.
    Frequency  FFT power-spectrum descriptors plus Welch mean frequency.
    Wavelet    Relative energy of each wavelet-packet node.
    Envelope   Descriptors of the Hilbert envelope (impulsiveness).

Cross-axis features (correlation, RMS ratio, energy ratio) are computed for
each pair of acceleration axes. Every window is tagged with the controller
``step_no`` closest in time to the window centre.

Several K values can be given at once: each cut is read from disk only once
and one CSV per split and K value is written.

Example
-------
    python extract_sensor_features.py \\
        --train-controller-dir Training_Dataset/Controller_Data \\
        --train-sensor-dir Training_Dataset/Sensor_Data \\
        --eval-controller-dir Evaluation_Dataset/Controller_Data \\
        --eval-sensor-dir Evaluation_Dataset/Sensor_Data \\
        --output-dir results/ \\
        -k 10
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import kurtosis, skew

from data_loader_evalset import DataLoader as EvalDataLoader
from Estancia2026.codigoLimpio.scripts.data_loader_trainset import DataLoader as TrainDataLoader

logger = logging.getLogger(__name__)

# Configuration

N_CUTS = 26
SENSOR_FS = 25600
WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
EPS = 1e-12

SENSOR_TIME_COL = "Date/Time"
CONTROLLER_TIME_COL = "timestamp"
CONTROLLER_STEP_COL = "step_no"
CONTROLLER_CUT_START_COL = "start_cut"
CONTROLLER_CUT_END_COL = "end_cut"

SENSOR_CHANNELS = (
    "Acceleration X (g)", "Acceleration Y (g)", "Acceleration Z (g)", "AE (V)",
)
CHANNEL_INDEX = {channel: i for i, channel in enumerate(SENSOR_CHANNELS)}
CROSS_AXIS_PAIRS = (
    ("XY", "Acceleration X (g)", "Acceleration Y (g)"),
    ("XZ", "Acceleration X (g)", "Acceleration Z (g)"),
    ("YZ", "Acceleration Y (g)", "Acceleration Z (g)"),
)

TEMPORAL_FEATURES = (
    "XMAX", "XMIN", "XMEAN", "XMEDIAN", "XRMS", "XSTD", "XSKEW", "XKURT", "XP2P",
    "XENTROPY", "XCREST", "XSHAPE", "XCLEARANCE", "XENERGY", "XIQR", "XPEAK_COUNT", "XZC",
)
FREQUENCY_FEATURES = (
    "SMAX", "SMEAN", "SVAR", "SSKEW", "SKURT", "SREL", "SCOG", "SSPREAD",
    "SENTROPY", "FPEAK", "SFLATNESS", "SKURT_SPECTRAL", "MEANFREQ", "SMED",
)
WAVELET_FEATURES = tuple(f"WPE_{i}" for i in range(2 ** WAVELET_LEVEL))
ENVELOPE_FEATURES = ("ENV_MEAN", "ENV_RMS", "ENV_STD", "ENV_KURT")
CROSS_AXIS_FEATURES = ("CORR", "RMS_RATIO", "ENERGY_RATIO")


def _nan_dict(names: Iterable[str]) -> dict[str, float]:
    return dict.fromkeys(names, np.nan)


# Feature functions
def temporal_features(x: np.ndarray) -> dict[str, float]:
    if x.size < 2:
        return _nan_dict(TEMPORAL_FEATURES)

    xmax = x.max()
    rms = np.sqrt(np.mean(x ** 2))
    abs_x = np.abs(x)
    abs_sum = abs_x.sum()
    mean_abs = abs_x.mean()
    q75, q25 = np.percentile(x, [75, 25])

    if abs_sum > 0:
        p = abs_x / abs_sum
        entropy = -np.sum(p * np.log(p + EPS))
    else:
        entropy = np.nan

    return {
        "XMAX": xmax,
        "XMIN": x.min(),
        "XMEAN": x.mean(),
        "XMEDIAN": np.median(x),
        "XRMS": rms,
        "XSTD": x.std(ddof=1),
        "XSKEW": skew(x),
        "XKURT": kurtosis(x),
        "XP2P": np.ptp(x),
        "XENTROPY": entropy,
        "XCREST": xmax / rms if rms != 0 else np.nan,
        "XSHAPE": rms / mean_abs if mean_abs != 0 else np.nan,
        "XCLEARANCE": xmax / (np.mean(np.sqrt(abs_x)) + EPS),
        "XENERGY": np.sum(x ** 2),
        "XIQR": q75 - q25,
        "XPEAK_COUNT": len(find_peaks(x)[0]),
        "XZC": int(np.count_nonzero(x[:-1] * x[1:] < 0)),
    }


def frequency_features(x: np.ndarray, fs: int = SENSOR_FS) -> dict[str, float]:
    n = x.size
    if n < 2:
        return _nan_dict(FREQUENCY_FEATURES)

    half = n // 2
    freqs = rfftfreq(n, 1 / fs)[:half]
    ps = np.abs(rfft(x)[:half]) ** 2 / n

    ps_sum = ps.sum()
    ps_mean = ps.mean()
    ps_var = ps.var()

    if ps_sum != 0:
        cog = np.sum(freqs * ps) / ps_sum
        spread = np.sqrt(np.sum((freqs - cog) ** 2 * ps) / ps_sum)
        ps_norm = ps / ps_sum
        entropy = -np.sum(ps_norm * np.log(ps_norm + EPS))
        rel = ps.max() / ps_sum
    else:
        cog = spread = entropy = rel = np.nan

    f_welch, pxx = welch(x, fs=fs, nperseg=min(1024, n))
    pxx_sum = pxx.sum()

    return {
        "SMAX": ps.max(),
        "SMEAN": ps_mean,
        "SVAR": ps_var,
        "SSKEW": skew(ps),
        "SKURT": kurtosis(ps),
        "SREL": rel,
        "SCOG": cog,
        "SSPREAD": spread,
        "SENTROPY": entropy,
        "FPEAK": freqs[np.argmax(ps)],
        "SFLATNESS": np.exp(np.mean(np.log(ps + EPS))) / (ps_mean + EPS),
        "SKURT_SPECTRAL": (
            np.mean((ps - ps_mean) ** 4) / (ps_var ** 2 + EPS) if ps_var > 0 else np.nan
        ),
        "MEANFREQ": np.sum(f_welch * pxx) / pxx_sum if pxx_sum != 0 else np.nan,
        "SMED": np.median(ps),
    }


def wavelet_packet_features(x: np.ndarray) -> dict[str, float]:
    if x.size < (2 ** WAVELET_LEVEL) * 4:
        return _nan_dict(WAVELET_FEATURES)

    wp = pywt.WaveletPacket(data=x, wavelet=WAVELET_NAME, mode="symmetric",
                            maxlevel=WAVELET_LEVEL)
    energies = np.array([np.sum(node.data ** 2)
                         for node in wp.get_level(WAVELET_LEVEL, order="freq")])
    total = energies.sum()
    if total <= 0:
        return _nan_dict(WAVELET_FEATURES)
    return dict(zip(WAVELET_FEATURES, energies / total))


def envelope_features(x: np.ndarray) -> dict[str, float]:
    if x.size < 2:
        return _nan_dict(ENVELOPE_FEATURES)

    env = np.abs(hilbert(x))
    return {
        "ENV_MEAN": env.mean(),
        "ENV_RMS": np.sqrt(np.mean(env ** 2)),
        "ENV_STD": env.std(ddof=1),
        "ENV_KURT": kurtosis(env),
    }


def cross_axis_features(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    valid = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[valid], b[valid]
    if a.size < 2:
        return _nan_dict(CROSS_AXIS_FEATURES)

    rms_a = np.sqrt(np.mean(a ** 2))
    rms_b = np.sqrt(np.mean(b ** 2))
    energy_a = np.sum(a ** 2)
    energy_total = energy_a + np.sum(b ** 2)

    return {
        "CORR": np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan,
        "RMS_RATIO": rms_a / rms_b if rms_b != 0 else np.nan,
        "ENERGY_RATIO": energy_a / energy_total if energy_total != 0 else np.nan,
    }


def window_features(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    row: dict[str, float] = {}

    for channel, i in CHANNEL_INDEX.items():
        x = values[:, i]
        x = x[~np.isnan(x)]
        for block in (temporal_features(x), frequency_features(x),
                      wavelet_packet_features(x), envelope_features(x)):
            for name, value in block.items():
                row[f"{name}_{channel}"] = value

    for label, ch_a, ch_b in CROSS_AXIS_PAIRS:
        feats = cross_axis_features(values[:, CHANNEL_INDEX[ch_a]], values[:, CHANNEL_INDEX[ch_b]])
        for name, value in feats.items():
            row[f"{label}_{name}"] = value

    return row


# Data preparation and windowing
def prepare_sensor(df: pd.DataFrame | None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return time-sorted sensor timestamps and a (n, n_channels) float32 matrix. 
    """
    empty = (pd.DatetimeIndex([]), np.empty((0, len(SENSOR_CHANNELS)), dtype=np.float32))
    if df is None or df.empty or SENSOR_TIME_COL not in df:
        return empty

    times = pd.to_datetime(df[SENSOR_TIME_COL], errors="coerce")
    valid = times.notna().to_numpy()
    if not valid.any():
        return empty

    times = pd.DatetimeIndex(times[valid])
    order = np.argsort(times.asi8, kind="stable")
    values = df.reindex(columns=list(SENSOR_CHANNELS)).to_numpy(dtype=np.float32)[valid]
    return times[order], values[order]


def prepare_controller(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Keep only the controller columns needed and parse their timestamps."""
    if df is None or df.empty:
        return None

    cols = [c for c in (CONTROLLER_TIME_COL, CONTROLLER_STEP_COL,
                        CONTROLLER_CUT_START_COL, CONTROLLER_CUT_END_COL) if c in df]
    df = df[cols].copy()
    for col in (CONTROLLER_TIME_COL, CONTROLLER_CUT_START_COL, CONTROLLER_CUT_END_COL):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def cut_time_bounds(
    sensor_times: pd.DatetimeIndex, controller: pd.DataFrame | None
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Start/end of a cut: controller cut markers if valid, else the data time span."""
    if controller is not None and {CONTROLLER_CUT_START_COL, CONTROLLER_CUT_END_COL} <= set(controller):
        starts = controller[CONTROLLER_CUT_START_COL].dropna()
        ends = controller[CONTROLLER_CUT_END_COL].dropna()
        if len(starts) and len(ends) and ends.iloc[0] > starts.iloc[0]:
            return starts.iloc[0], ends.iloc[0]

    candidates = []
    if controller is not None and CONTROLLER_TIME_COL in controller:
        t = controller[CONTROLLER_TIME_COL].dropna()
        if len(t):
            candidates += [t.min(), t.max()]
    if len(sensor_times):
        candidates += [sensor_times[0], sensor_times[-1]]

    if not candidates:
        return None
    start, end = min(candidates), max(candidates)
    return (start, end) if end > start else None


def assign_steps(edges: pd.DatetimeIndex, controller: pd.DataFrame | None) -> np.ndarray:
    """``step_no`` of the controller record closest in time to each window centre."""
    k = len(edges) - 1
    if (controller is None or CONTROLLER_TIME_COL not in controller
            or CONTROLLER_STEP_COL not in controller):
        return np.full(k, np.nan)

    ctrl = (controller[[CONTROLLER_TIME_COL, CONTROLLER_STEP_COL]]
            .dropna(subset=[CONTROLLER_TIME_COL])
            .sort_values(CONTROLLER_TIME_COL))
    if ctrl.empty:
        return np.full(k, np.nan)

    centres = pd.DataFrame({"t": edges[:-1] + (edges[1:] - edges[:-1]) / 2})
    merged = pd.merge_asof(centres, ctrl, left_on="t", right_on=CONTROLLER_TIME_COL,
                           direction="nearest")
    return merged[CONTROLLER_STEP_COL].to_numpy()


def cut_rows(
    dataset: int,
    cut: int,
    k: int,
    sensor_times: pd.DatetimeIndex,
    sensor_values: np.ndarray,
    controller: pd.DataFrame | None,
    bounds: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> list[dict]:
    """Feature rows for the K windows of one cut."""
    if bounds is None:
        steps = np.full(k, np.nan)
        slices = [slice(0, 0)] * k
    else:
        edges = pd.date_range(*bounds, periods=k + 1)
        steps = assign_steps(edges, controller)
        if len(sensor_times):
            # Windows are [start, end), except the last one which is [start, end].
            starts = sensor_times.searchsorted(edges, side="left")
            ends = np.append(starts[1:-1], sensor_times.searchsorted(edges[-1], side="right"))
            slices = [slice(s, e) for s, e in zip(starts[:-1], ends)]
        else:
            slices = [slice(0, 0)] * k

    return [
        {"dataset": dataset, "cut": cut, "window": w, "step": steps[w],
         **window_features(sensor_values[slices[w]])}
        for w in range(k)
    ]


# Pipeline
def _load(fetch, kind: str, dataset: int, cut: int) -> pd.DataFrame | None:
    try:
        return fetch(dataset, cut)
    except Exception as exc:  # loader-specific errors
        logger.warning("Could not load %s data for dataset %d, cut %d (%s).",
                       kind, dataset, cut, exc)
        return None


def extract_features(
    loader, dataset_ids: Iterable[int], k_values: list[int]
) -> dict[int, pd.DataFrame]:
    """Extract window features for every K, reading each cut from disk only once."""
    rows: dict[int, list[dict]] = {k: [] for k in k_values}

    for dataset in dataset_ids:
        for cut in range(1, N_CUTS + 1):
            logger.info("Dataset %d, cut %d", dataset, cut)
            sensor_times, sensor_values = prepare_sensor(
                _load(loader.get_sensor_data, "sensor", dataset, cut))
            controller = prepare_controller(
                _load(loader.get_controller_data, "controller", dataset, cut))
            bounds = cut_time_bounds(sensor_times, controller)

            for k in k_values:
                rows[k].extend(cut_rows(dataset, cut, k, sensor_times, sensor_values,
                                        controller, bounds))

    return {k: pd.DataFrame(r) for k, r in rows.items()}


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"K must be a positive integer, got {value}.")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract windowed features from CNC sensor data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-controller-dir", type=Path, required=True,
                        help="Directory with the training controller data.")
    parser.add_argument("--train-sensor-dir", type=Path, required=True,
                        help="Directory with the training sensor data.")
    parser.add_argument("--eval-controller-dir", type=Path, required=True,
                        help="Directory with the evaluation controller data.")
    parser.add_argument("--eval-sensor-dir", type=Path, required=True,
                        help="Directory with the evaluation sensor data.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where the feature CSV files are written.")
    parser.add_argument("-k", "--windows", dest="k", type=positive_int, nargs="+", required=True,
                        help="Number of time windows per cut. Several values may be given.")
    parser.add_argument("--train-datasets", type=int, nargs="+", default=list(range(1, 7)),
                        help="Training dataset IDs to process.")
    parser.add_argument("--eval-datasets", type=int, nargs="+", default=list(range(1, 4)),
                        help="Evaluation dataset IDs to process.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only log warnings and errors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    k_values = sorted(set(args.k))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (TrainDataLoader(str(args.train_controller_dir), str(args.train_sensor_dir)),
                  args.train_datasets),
        "eval": (EvalDataLoader(str(args.eval_controller_dir), str(args.eval_sensor_dir)),
                 args.eval_datasets),
    }

    for split, (loader, dataset_ids) in splits.items():
        start = time.perf_counter()
        features = extract_features(loader, dataset_ids, k_values)
        logger.info("[%s] Features extracted in %.2f s.", split, time.perf_counter() - start)

        for k, df in features.items():
            path = args.output_dir / f"features_sensor_{split}_K{k}.csv"
            df.to_csv(path, index=False)
            logger.info("[%s] K=%d: saved %d rows x %d columns to %s",
                        split, k, len(df), df.shape[1], path)


if __name__ == "__main__":
    main()