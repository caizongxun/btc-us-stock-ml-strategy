"""N-of-M signal voting — combine multiple signal sources"""
import pandas as pd
import numpy as np
from config import CFG

def build_multi_signal(
    df: pd.DataFrame,
    model_signal: pd.Series,
    n_required: int = CFG.n_of_m_threshold,
) -> pd.Series:
    """
    Combine ML signal with rule-based conditions.
    Each sub-signal = +1 (long) or -1 (not long).
    Final signal fires +1 only if n_required conditions agree.
    """
    sub_signals = {}

    # 1. ML model signal
    sub_signals["ml_signal"] = (model_signal == 1).astype(int)

    # 2. BTC trend score >= 2 (at least 2 of 3 EMAs bullish)
    if "btc_trend_score" in df.columns:
        sub_signals["trend_bullish"] = (df["btc_trend_score"] >= 2).astype(int)

    # 3. VIX low regime (risk-on)
    if "vix_low" in df.columns:
        sub_signals["vix_low"] = df["vix_low"].astype(int)

    # 4. BTC not in bear regime
    if "btc_regime_bear" in df.columns:
        sub_signals["not_bear_regime"] = (1 - df["btc_regime_bear"]).astype(int)

    # 5. Fear & Greed not extreme greed (avoid FOMO tops)
    if "fg_value" in df.columns:
        sub_signals["fg_not_extreme_greed"] = (df["fg_value"] < 75).astype(int)

    # 6. BTC-SPY correlation risk-on
    if "btc_SPY_corr_30d" in df.columns:
        sub_signals["risk_on_corr"] = (df["btc_SPY_corr_30d"] > 0.3).astype(int)

    # 7. BTC RSI not overbought
    if "btc_rsi14_ob" in df.columns:
        sub_signals["rsi_not_ob"] = (1 - df["btc_rsi14_ob"]).astype(int)

    # 8. Volume surge (momentum confirmation)
    if "btc_vol_ma_ratio_7d" in df.columns:
        sub_signals["vol_surge"] = (df["btc_vol_ma_ratio_7d"] > 1.2).astype(int)

    signal_df = pd.DataFrame(sub_signals, index=df.index)
    vote_count = signal_df.sum(axis=1)
    final_signal = pd.Series(0, index=df.index)
    final_signal[vote_count >= n_required] = 1
    final_signal[vote_count < (len(sub_signals) - n_required)] = -1

    return final_signal, signal_df
