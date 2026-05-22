"""N-of-M signal voting — combine ML signal with rule-based conditions.

Design principle:
  Only generate LONG (+1) or FLAT (0) signals.
  NO short (-1) signal — BTC has positive drift; short bias destroys returns
  in bull periods and the model AUC (~0.57) is not strong enough to time shorts.

  Short/flat distinction:
    vote_count >= n_required  -> LONG
    vote_count <  n_required  -> FLAT  (stay out, not short)

Sub-signal thresholds calibrated against actual vote count distributions:
  risk_on_corr : corr > 0.1  (was 0.3 — too few fires at 0.3)
  risk_on_score: score > 0.4  (was 0.5)
  vol_surge    : ratio > 1.1  (was 1.2)
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
    Returns final_signal (+1 long, 0 flat) and signal_df (component breakdown).
    """
    sub_signals: dict = {}

    # 1. ML model signal
    sub_signals["ml_signal"] = (model_signal == 1).astype(int)

    # 2. BTC trend score >= 2 (at least 2 of 3 EMAs aligned bullish)
    if "btc_trend_score" in df.columns:
        sub_signals["trend_bullish"] = (df["btc_trend_score"] >= 2).astype(int)
    else:
        logger.warning("btc_trend_score not found")

    # 3. BTC not in bear regime
    if "btc_regime_bear" in df.columns:
        sub_signals["not_bear_regime"] = (1 - df["btc_regime_bear"]).astype(int)
    else:
        logger.warning("btc_regime_bear not found")

    # 4. BTC not in sideways regime
    if "btc_regime_sideways" in df.columns:
        sub_signals["not_sideways"] = (1 - df["btc_regime_sideways"]).astype(int)
    else:
        logger.warning("btc_regime_sideways not found")

    # 5. Fear & Greed not extreme greed (<75)
    if "fg_value" in df.columns:
        sub_signals["fg_not_extreme_greed"] = (df["fg_value"] < 75).astype(int)
    else:
        logger.warning("fg_value not found")

    # 6. BTC-SPY correlation risk-on (relaxed: > 0.1 instead of 0.3)
    corr_col = "btc_SPY_corr_30d"
    if corr_col in df.columns:
        sub_signals["risk_on_corr"] = (df[corr_col] > 0.1).astype(int)
    else:
        logger.warning(f"{corr_col} not found")

    # 7. BTC RSI14 not overbought
    if "btc_rsi14_ob" in df.columns:
        sub_signals["rsi_not_ob"] = (1 - df["btc_rsi14_ob"]).astype(int)
    else:
        logger.warning("btc_rsi14_ob not found")

    # 8. Volume surge confirmation (relaxed: > 1.1 instead of 1.2)
    vol_col = (
        "btc_vol_ma_ratio_14d" if "btc_vol_ma_ratio_14d" in df.columns
        else "btc_vol_ma_ratio_7d"
    )
    if vol_col in df.columns:
        sub_signals["vol_surge"] = (df[vol_col] > 1.1).astype(int)
    else:
        logger.warning("btc_vol_ma_ratio_*d not found")

    # 9. Cross-asset risk-on composite (relaxed: > 0.4 instead of 0.5)
    if "risk_on_score" in df.columns:
        sub_signals["risk_on_score"] = (df["risk_on_score"] > 0.4).astype(int)
    else:
        logger.warning("risk_on_score not found")

    # ------------------------------------------------------------------
    signal_df = pd.DataFrame(sub_signals, index=df.index)
    n_total = len(sub_signals)

    if n_required is None:
        n_required = CFG.n_of_m_threshold  # default from config (now 4)
        # Floor: always require at least ceil(n_total/2) to avoid noise
        n_required = max(n_required, (n_total + 1) // 2)

    vote_counts = {k: int(v.sum()) for k, v in sub_signals.items()}
    logger.info(
        f"multi_signal: {n_total} sub-signals, requiring {n_required} to fire.\n"
        f"  Vote counts: {vote_counts}"
    )

    vote_count = signal_df.sum(axis=1)

    # LONG only when enough signals agree; otherwise FLAT (never forced short)
    final_signal = pd.Series(0, index=df.index)
    final_signal[vote_count >= n_required] = 1

    pos = (final_signal == 1).sum()
    flat = (final_signal == 0).sum()
    logger.info(f"  Final signal: long={pos}, flat={flat}  "
                f"(long rate={100*pos/len(final_signal):.1f}%)")

    return final_signal, signal_df
