"""N-of-M signal voting — combine ML signal with rule-based conditions.

Design:
  - LONG (+1) or FLAT (0) only. No forced short.
  - Base threshold: 5-of-9 sub-signals must agree.
  - ML boost: if ml_signal fires, effective threshold drops to 4-of-9,
    because the ML model has already synthesised all 50 features.
  - Threshold calibration (vs observed vote counts ~1193 bars):
      btc_trend_score >= 2  : ~706 fires (59%)
      not_bear_regime       : ~810 fires (68%)
      not_sideways          : ~734 fires (62%)
      fg_not_extreme_greed  : ~923 fires (77%)
      rsi_not_ob            : ~972 fires (82%)
      vol_surge (>1.1)      : ~407 fires (34%)
      risk_on_corr (>0.0)   : ~500+ fires after relaxation
      risk_on_score (>0.35) : ~500+ fires after relaxation
      ml_signal (>0.45)     : ~100-200 fires after threshold fix
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
    sub_signals: dict = {}

    # 1. ML model signal (threshold lowered to 0.45 in config)
    sub_signals["ml_signal"] = (model_signal == 1).astype(int)

    # 2. BTC EMA trend alignment >= 2-of-3
    if "btc_trend_score" in df.columns:
        sub_signals["trend_bullish"] = (df["btc_trend_score"] >= 2).astype(int)
    else:
        logger.warning("btc_trend_score not found")

    # 3. Not in bear regime
    if "btc_regime_bear" in df.columns:
        sub_signals["not_bear_regime"] = (1 - df["btc_regime_bear"]).astype(int)
    else:
        logger.warning("btc_regime_bear not found")

    # 4. Not in sideways regime
    if "btc_regime_sideways" in df.columns:
        sub_signals["not_sideways"] = (1 - df["btc_regime_sideways"]).astype(int)
    else:
        logger.warning("btc_regime_sideways not found")

    # 5. Fear & Greed not extreme greed (<75)
    if "fg_value" in df.columns:
        sub_signals["fg_not_extreme_greed"] = (df["fg_value"] < 75).astype(int)
    else:
        logger.warning("fg_value not found")

    # 6. BTC-SPY positive correlation (any positive = risk-on, relaxed from 0.3->0.0)
    corr_col = "btc_SPY_corr_30d"
    if corr_col in df.columns:
        sub_signals["risk_on_corr"] = (df[corr_col] > 0.0).astype(int)
    else:
        logger.warning(f"{corr_col} not found")

    # 7. RSI14 not overbought
    if "btc_rsi14_ob" in df.columns:
        sub_signals["rsi_not_ob"] = (1 - df["btc_rsi14_ob"]).astype(int)
    else:
        logger.warning("btc_rsi14_ob not found")

    # 8. Volume surge (ratio > 1.1)
    vol_col = (
        "btc_vol_ma_ratio_14d" if "btc_vol_ma_ratio_14d" in df.columns
        else "btc_vol_ma_ratio_7d"
    )
    if vol_col in df.columns:
        sub_signals["vol_surge"] = (df[vol_col] > 1.1).astype(int)
    else:
        logger.warning("btc_vol_ma_ratio_*d not found")

    # 9. Cross-asset risk-on score > 0.35 (relaxed from 0.5->0.35)
    if "risk_on_score" in df.columns:
        sub_signals["risk_on_score"] = (df["risk_on_score"] > 0.35).astype(int)
    else:
        logger.warning("risk_on_score not found")

    # ------------------------------------------------------------------
    signal_df = pd.DataFrame(sub_signals, index=df.index)
    n_total = len(sub_signals)

    base_required = n_required if n_required is not None else CFG.n_of_m_threshold
    # Safety floor: at least ceil(n_total/2)
    base_required = max(base_required, (n_total + 1) // 2)

    vote_counts = {k: int(v.sum()) for k, v in sub_signals.items()}
    logger.info(
        f"multi_signal: {n_total} sub-signals, base_required={base_required}.\n"
        f"  Vote counts: {vote_counts}"
    )

    vote_count = signal_df.sum(axis=1)
    ml_fires = signal_df.get("ml_signal", pd.Series(0, index=df.index))

    # ML boost: when ML agrees, lower threshold by 1 (ML already encodes 50 features)
    effective_required = base_required - ml_fires  # Series: base or base-1

    final_signal = pd.Series(0, index=df.index)
    final_signal[vote_count >= effective_required] = 1

    pos  = (final_signal == 1).sum()
    flat = (final_signal == 0).sum()
    ml_boost_count = int((ml_fires == 1).sum())
    logger.info(
        f"  Final signal: long={pos}, flat={flat}  (long rate={100*pos/len(final_signal):.1f}%)\n"
        f"  ML boost applied to {ml_boost_count} bars"
    )

    return final_signal, signal_df
