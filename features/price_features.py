"""Price-based features: returns, momentum, drawdown, position in range"""
import pandas as pd
import numpy as np
from config import CFG

def add_price_features(df: pd.DataFrame, prefix: str = "btc") -> pd.DataFrame:
    """
    Input: df with columns [open, high, low, close, volume]
    All features are stationary (returns / ratios / z-scores) — no raw prices.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # --- Returns ---
    for n in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]:
        df[f"{prefix}_ret_{n}d"] = c.pct_change(n)

    # --- Log returns ---
    for n in [1, 3, 5, 14]:
        df[f"{prefix}_logret_{n}d"] = np.log(c / c.shift(n))

    # --- Candle body ---
    df[f"{prefix}_body_ratio"] = (c - o) / (h - l + 1e-9)  # -1 to +1
    df[f"{prefix}_upper_wick"] = (h - c.clip(lower=o)) / (h - l + 1e-9)
    df[f"{prefix}_lower_wick"] = (c.clip(upper=o) - l) / (h - l + 1e-9)

    # --- High/Low position ---
    for n in [14, 30, 60]:
        roll_h = h.rolling(n).max()
        roll_l = l.rolling(n).min()
        df[f"{prefix}_hl_pos_{n}d"] = (c - roll_l) / (roll_h - roll_l + 1e-9)

    # --- Drawdown from rolling high ---
    for n in [30, 90]:
        roll_h = c.rolling(n).max()
        df[f"{prefix}_dd_{n}d"] = (c - roll_h) / (roll_h + 1e-9)

    # --- Volume features ---
    for n in [7, 14, 30]:
        df[f"{prefix}_vol_ma_ratio_{n}d"] = v / (v.rolling(n).mean() + 1e-9)
    df[f"{prefix}_vol_ret_corr"] = c.pct_change().rolling(14).corr(v.pct_change())

    # --- Z-score of price vs rolling mean ---
    for n in [20, 60]:
        m = c.rolling(n).mean()
        s = c.rolling(n).std()
        df[f"{prefix}_zscore_{n}d"] = (c - m) / (s + 1e-9)

    # --- Gap (open vs prior close) ---
    df[f"{prefix}_gap"] = (o - c.shift(1)) / (c.shift(1) + 1e-9)

    return df
