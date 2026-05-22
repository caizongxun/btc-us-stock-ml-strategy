"""N-of-M voting strategy — LONG or FLAT only.

Sub-signal design target: each signal fires ~25-60% of the time.
Signals firing >75% or <15% are weak discriminators and replaced.

Final architecture (10 sub-signals, require 6-of-10):
  Signal                 | Target fire rate | Source
  ---------------------- | ---------------- | ------
  ml_signal_top20        | ~20%             | XGB rolling top-20% prob rank
  trend_bullish          | ~59%             | EMA alignment >= 2-of-3
  not_bear_regime        | ~68%             | HMM regime != bear
  fg_not_extreme         | ~55%             | Fear&Greed 25 < val < 75 (both extremes excluded)
  risk_on_corr_strong    | ~40%             | BTC-SPY 30d corr > 0.35
  risk_on_score_hi       | ~50%             | risk_on_score > 0.55 (above median)
  vol_surge              | ~34%             | vol_ma_ratio_14d > 1.1
  btc_above_ema200       | ~55%             | close > EMA200 (trend filter)
  gld_div_negative       | ~45%             | btc_GLD_div_5d < 0.03 (BTC not overbought vs gold)
  vix_low                | ~45%             | VIX < 20

ML boost: when ml_signal_top20 fires, effective threshold drops 6 -> 5.
"""
import pandas as pd
import numpy as np
from loguru import logger
from config import CFG


def build_multi_signal(
    df: pd.DataFrame,
    model_signal: pd.Series,           # +1/0/-1 from model predict
    model_prob: pd.Series | None = None,  # raw long probability (optional)
    n_required: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns (final_signal, signal_df).
    final_signal: +1 long, 0 flat.
    signal_df: component breakdown DataFrame.
    """
    sub_signals: dict = {}

    # ------------------------------------------------------------------
    # 1. ML signal — dynamic top-20% rolling rank (60-bar window)
    #    Avoids fixed-threshold problem when calibrated prob is compressed.
    # ------------------------------------------------------------------
    if model_prob is not None and model_prob.notna().sum() > 30:
        # Rolling 60-bar rank: 1 = top 20% predictions
        roll_rank = model_prob.rolling(60, min_periods=20).rank(pct=True)
        sub_signals["ml_signal_top20"] = (roll_rank >= 0.80).astype(int)
    else:
        # Fallback: use model_signal directly
        sub_signals["ml_signal_top20"] = (model_signal == 1).astype(int)

    # ------------------------------------------------------------------
    # 2. BTC EMA trend >= 2-of-3 EMAs aligned bullish (~59%)
    # ------------------------------------------------------------------
    if "btc_trend_score" in df.columns:
        sub_signals["trend_bullish"] = (df["btc_trend_score"] >= 2).astype(int)
    else:
        logger.warning("btc_trend_score not found")

    # ------------------------------------------------------------------
    # 3. Not in bear regime (~68%)
    # ------------------------------------------------------------------
    if "btc_regime_bear" in df.columns:
        sub_signals["not_bear_regime"] = (1 - df["btc_regime_bear"]).astype(int)
    else:
        logger.warning("btc_regime_bear not found")

    # ------------------------------------------------------------------
    # 4. Fear & Greed in neutral zone 25-75 (~55%)
    #    Excludes BOTH extreme fear AND extreme greed
    # ------------------------------------------------------------------
    if "fg_value" in df.columns:
        sub_signals["fg_neutral"] = (
            (df["fg_value"] > 25) & (df["fg_value"] < 75)
        ).astype(int)
    else:
        logger.warning("fg_value not found")

    # ------------------------------------------------------------------
    # 5. BTC-SPY 30d corr > 0.35 — strong risk-on alignment (~40%)
    #    (was > 0.0 = always fires; now meaningful threshold)
    # ------------------------------------------------------------------
    corr_col = "btc_SPY_corr_30d"
    if corr_col in df.columns:
        sub_signals["risk_on_corr_strong"] = (df[corr_col] > 0.35).astype(int)
    else:
        logger.warning(f"{corr_col} not found")

    # ------------------------------------------------------------------
    # 6. Risk-on composite score above median (> 0.55) (~50%)
    # ------------------------------------------------------------------
    if "risk_on_score" in df.columns:
        sub_signals["risk_on_score_hi"] = (df["risk_on_score"] > 0.55).astype(int)
    else:
        logger.warning("risk_on_score not found")

    # ------------------------------------------------------------------
    # 7. Volume surge — btc vol > 14d moving average * 1.1 (~34%)
    # ------------------------------------------------------------------
    vol_col = (
        "btc_vol_ma_ratio_14d" if "btc_vol_ma_ratio_14d" in df.columns
        else "btc_vol_ma_ratio_7d"
    )
    if vol_col in df.columns:
        sub_signals["vol_surge"] = (df[vol_col] > 1.1).astype(int)
    else:
        logger.warning("btc_vol_ma_ratio_*d not found")

    # ------------------------------------------------------------------
    # 8. BTC above EMA200 — primary trend filter (~55%)
    #    Uses btc_price_vs_ema200 (ratio) or btc_ema200_slope as proxy
    # ------------------------------------------------------------------
    if "btc_price_vs_ema200" in df.columns:
        sub_signals["btc_above_ema200"] = (df["btc_price_vs_ema200"] > 0).astype(int)
    elif "btc_ema200_slope" in df.columns:
        sub_signals["btc_above_ema200"] = (df["btc_ema200_slope"] > 0).astype(int)
    else:
        logger.warning("btc_price_vs_ema200 / btc_ema200_slope not found")

    # ------------------------------------------------------------------
    # 9. BTC-GLD divergence < 0.03 — BTC not overextended vs gold (~45%)
    #    From DT rules: btc_GLD_div_5d is top SHAP feature
    # ------------------------------------------------------------------
    gld_col = "btc_GLD_div_5d"
    if gld_col in df.columns:
        sub_signals["gld_div_ok"] = (df[gld_col] < 0.03).astype(int)
    else:
        logger.warning(f"{gld_col} not found")

    # ------------------------------------------------------------------
    # 10. VIX low — broad market fear low (~45%)
    # ------------------------------------------------------------------
    if "vix_low" in df.columns:
        sub_signals["vix_low"] = df["vix_low"].astype(int)
    elif "vix" in df.columns:
        sub_signals["vix_low"] = (df["vix"] < 20).astype(int)
    else:
        logger.warning("vix_low / vix not found")

    # ------------------------------------------------------------------
    signal_df = pd.DataFrame(sub_signals, index=df.index)
    n_total = len(sub_signals)

    base_required = n_required if n_required is not None else CFG.n_of_m_threshold
    # floor at ceil(n_total * 0.55) so long rate stays ~30-40%
    floor = max(5, int(np.ceil(n_total * 0.55)))
    base_required = max(base_required, floor)

    vote_counts = {k: int(v.sum()) for k, v in sub_signals.items()}
    fire_rates  = {k: f"{100*v.mean():.0f}%" for k, v in sub_signals.items()}
    logger.info(
        f"multi_signal: {n_total} sub-signals, base_required={base_required}\n"
        f"  Fire rates : {fire_rates}\n"
        f"  Vote counts: {vote_counts}"
    )

    vote_count = signal_df.sum(axis=1)
    ml_fires   = signal_df.get("ml_signal_top20", pd.Series(0, index=df.index))

    # ML boost: top-20% ML prediction lowers bar by 1
    effective_required = base_required - ml_fires

    final_signal = pd.Series(0, index=df.index)
    final_signal[vote_count >= effective_required] = 1

    pos  = int((final_signal == 1).sum())
    flat = int((final_signal == 0).sum())
    ml_boost_count = int((ml_fires == 1).sum())
    logger.info(
        f"  Final signal: long={pos}, flat={flat}  "
        f"(long rate={100*pos/len(final_signal):.1f}%)\n"
        f"  ML boost applied to {ml_boost_count} bars"
    )

    return final_signal, signal_df
