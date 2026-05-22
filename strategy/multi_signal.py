"""N-of-M signal voting — combine ML signal with rule-based conditions.

Fix: add proper logger import; dynamic n_required; detailed vote logging.
"""
import pandas as pd
import numpy as np
from loguru import logger
from config import CFG


def build_multi_signal(
    df: pd.DataFrame,
    model_signal: pd.Series,
    n_required: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Combine ML signal with rule-based conditions.
    Each sub-signal = +1 (long) or 0 (not long).
    Final signal fires:
      +1  if vote_count >= n_required
      -1  if vote_count < (total - n_required)  [strong disagreement]
       0  otherwise
    """
    sub_signals: dict = {}

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
    n_total = len(sub_signals)

    # Dynamic n_required
    if n_required is None:
        if n_total >= 6:
            n_required = CFG.n_of_m_threshold   # default = 3
        elif n_total >= 4:
            n_required = 2
        else:
            n_required = 1

    vote_counts = {k: int(v.sum()) for k, v in sub_signals.items()}
    logger.info(
        f"multi_signal: {n_total} sub-signals, requiring {n_required} to fire.\n"
        f"  Vote counts: {vote_counts}"
    )

    vote_count = signal_df.sum(axis=1)
    final_signal = pd.Series(0, index=df.index)
    final_signal[vote_count >= n_required] = 1
    final_signal[vote_count < (n_total - n_required)] = -1

    pos = (final_signal == 1).sum()
    neg = (final_signal == -1).sum()
    logger.info(f"  Final signal: long={pos}, short={neg}, flat={len(final_signal)-pos-neg}")

    return final_signal, signal_df
