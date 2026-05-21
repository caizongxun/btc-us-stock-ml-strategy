"""Volatility features: ATR, realized vol, vol regime, VIX-derived"""
import pandas as pd
import numpy as np
from config import CFG

def _atr(high, low, close, n):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_volatility_features(df: pd.DataFrame, prefix: str = "btc") -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    ret = c.pct_change()

    # --- ATR (normalized) ---
    for n in [7, 14, 21, 30]:
        atr = _atr(h, l, c, n)
        df[f"{prefix}_atr_{n}d"] = atr / (c + 1e-9)  # normalize

    # --- Realized volatility (annualized) ---
    for n in [7, 14, 30, 60]:
        df[f"{prefix}_rvol_{n}d"] = ret.rolling(n).std() * np.sqrt(252)

    # --- Vol-of-vol ---
    df[f"{prefix}_vol_of_vol"] = df[f"{prefix}_rvol_14d"].rolling(20).std()

    # --- Vol ratio (short / long) — vol expansion signal ---
    df[f"{prefix}_vol_ratio_7_30"] = df[f"{prefix}_rvol_7d"] / (df[f"{prefix}_rvol_30d"] + 1e-9)
    df[f"{prefix}_vol_ratio_14_60"] = df[f"{prefix}_rvol_14d"] / (df[f"{prefix}_rvol_60d"] + 1e-9)

    # --- Parkinson volatility ---
    df[f"{prefix}_park_vol"] = (np.log(h / l) ** 2 / (4 * np.log(2))).rolling(14).mean() ** 0.5 * np.sqrt(252)

    # --- High volatility regime flag ---
    rv30 = df[f"{prefix}_rvol_30d"]
    df[f"{prefix}_high_vol_regime"] = (rv30 > rv30.rolling(90).quantile(0.75)).astype(int)
    df[f"{prefix}_low_vol_regime"] = (rv30 < rv30.rolling(90).quantile(0.25)).astype(int)

    return df

def add_vix_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """VIX-derived features added as separate columns."""
    v = vix_df["close"].rename("vix")
    out = pd.DataFrame(index=vix_df.index)
    out["vix"] = v
    out["vix_ma20"] = v.rolling(20).mean()
    out["vix_zscore"] = (v - v.rolling(60).mean()) / (v.rolling(60).std() + 1e-9)
    out["vix_spike"] = (v > v.shift(1) * 1.15).astype(int)      # >15% jump in VIX
    out["vix_low"] = (v < 18).astype(int)
    out["vix_high"] = (v > 30).astype(int)
    out["vix_ret_1d"] = v.pct_change(1)
    out["vix_ret_5d"] = v.pct_change(5)
    return out
