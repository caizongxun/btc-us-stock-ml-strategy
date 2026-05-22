"""Convert model probabilities -> +1/0/-1 trading signal"""
import numpy as np
import pandas as pd
from config import CFG

_DEFAULT_THRESHOLD = 0.45  # fallback when CFG.long_prob_threshold is None


def _resolve_threshold(value) -> float:
    """Return value if it is a float, else fall back to CFG or 0.45."""
    if value is not None:
        return float(value)
    if CFG.long_prob_threshold is not None:
        return float(CFG.long_prob_threshold)
    return _DEFAULT_THRESHOLD


def generate_signal_from_prob(
    prob: np.ndarray,
    long_thresh=None,
    short_thresh=None,
) -> np.ndarray:
    """
    prob        : array of P(long) from classifier
    long_thresh : override threshold (float).  None → resolve from CFG / fallback.
    short_thresh: override threshold (float).  None → resolve from CFG / fallback.

    Returns: array of signals  +1 (long) / 0 (flat) / -1 (short)
    """
    lt = _resolve_threshold(long_thresh)
    st = _resolve_threshold(short_thresh)

    signal = np.zeros(len(prob), dtype=int)
    signal[prob >= lt] = 1
    signal[prob <= (1.0 - st)] = -1   # short when prob of long is very low
    return signal


def generate_signal_from_rules(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """
    Apply extracted DT thresholds as a hard binary rule.
    Returns Series of +1/0/-1.
    """
    signal = pd.Series(0, index=df.index)
    conditions = []
    for feat, thresh_list in thresholds.items():
        if feat not in df.columns:
            continue
        t = thresh_list[0]  # use first threshold
        conditions.append(df[feat] > t)

    if conditions:
        from functools import reduce
        import operator
        combined = reduce(operator.and_, conditions)
        signal[combined] = 1
        signal[~combined] = -1
    return signal
