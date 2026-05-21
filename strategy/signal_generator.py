"""Convert model probabilities -> +1/0/-1 trading signal"""
import numpy as np
import pandas as pd
from config import CFG

def generate_signal_from_prob(
    prob: np.ndarray,
    long_thresh: float = CFG.long_prob_threshold,
    short_thresh: float = CFG.short_prob_threshold,
) -> np.ndarray:
    """
    prob: array of P(long) from classifier
    Returns: array of signals: +1, 0, -1
    """
    signal = np.zeros(len(prob), dtype=int)
    signal[prob >= long_thresh] = 1
    signal[prob <= short_thresh] = -1
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
