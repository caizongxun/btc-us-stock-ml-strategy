"""Cross-asset features: BTC-SPY correlation, beta, divergence, risk-on score.

IMPORTANT: btc_df passed here must be DAILY resampled (not 4h raw).
If 4h data is passed, rolling(30) covers only 5 calendar days,
making correlation values meaningless.
See build_features._resample_btc_daily().
"""
import pandas as pd
import numpy as np
from loguru import logger


def add_cross_asset_features(
    btc_df: pd.DataFrame, stock_dfs: dict
) -> pd.DataFrame:
    """
    btc_df  : daily BTC OHLCV DataFrame
    stock_dfs: dict of symbol -> daily OHLCV DataFrame (^VIX excluded)
    Returns : feature DataFrame on daily index
    """
    # Guard: warn if index looks intraday (more than 2x expected daily rows)
    expected_daily_rows = 365 * 4  # 4 years
    if len(btc_df) > expected_daily_rows * 6:
        logger.warning(
            f"cross_asset_features received {len(btc_df)} rows — "
            "looks like intraday data. Corr windows will be wrong. "
            "Pass daily-resampled BTC instead."
        )

    btc_ret = btc_df["close"].pct_change().rename("btc_ret")
    out = pd.DataFrame(index=btc_df.index)

    for sym, sdf in stock_dfs.items():
        s_ret = sdf["close"].pct_change().rename(f"{sym}_ret")
        aligned = pd.concat([btc_ret, s_ret], axis=1, join="inner")
        s_col = f"{sym}_ret"

        # Rolling correlation — min_periods=70% of window to handle gaps
        for w in [14, 30, 60]:
            min_p = max(5, int(w * 0.7))
            col = f"btc_{sym}_corr_{w}d"
            out[col] = (
                aligned["btc_ret"]
                .rolling(w, min_periods=min_p)
                .corr(aligned[s_col])
                .reindex(out.index)
            )

        # Rolling beta
        for w in [30, 60]:
            min_p = max(5, int(w * 0.7))
            cov = (
                aligned["btc_ret"]
                .rolling(w, min_periods=min_p)
                .cov(aligned[s_col])
                .reindex(out.index)
            )
            var = (
                aligned[s_col]
                .rolling(w, min_periods=min_p)
                .var()
                .reindex(out.index)
            )
            out[f"btc_{sym}_beta_{w}d"] = cov / (var + 1e-9)

        # Divergence: BTC 5d return minus stock 5d return
        btc_5d = btc_ret.rolling(5, min_periods=3).sum()
        s_5d   = s_ret.rolling(5, min_periods=3).sum()
        out[f"btc_{sym}_div_5d"]  = (btc_5d - s_5d.reindex(btc_5d.index)).reindex(out.index).fillna(0)
        out[f"btc_{sym}_div_14d"] = (
            btc_ret.rolling(14, min_periods=8).sum()
            - s_ret.rolling(14, min_periods=8).sum().reindex(btc_ret.index).fillna(0)
        ).reindex(out.index)

        # Lead-lag
        out[f"{sym}_lag1_ret"] = s_ret.shift(1).reindex(out.index)
        out[f"{sym}_lag2_ret"] = s_ret.shift(2).reindex(out.index)

        # Stock return features
        for n in [1, 3, 5, 14]:
            out[f"{sym}_ret_{n}d"] = s_ret.rolling(n, min_periods=1).sum().reindex(out.index)

    # Risk-on composite score — percentile rank 0~1
    if "SPY" in stock_dfs and "QQQ" in stock_dfs:
        spy_ret_5d = stock_dfs["SPY"]["close"].pct_change(5).reindex(out.index)
        qqq_ret_5d = stock_dfs["QQQ"]["close"].pct_change(5).reindex(out.index)
        btc_5d_r   = btc_df["close"].pct_change(5).reindex(out.index)
        out["risk_on_score"] = (
            spy_ret_5d.rank(pct=True) * 0.3
            + qqq_ret_5d.rank(pct=True) * 0.3
            + btc_5d_r.rank(pct=True) * 0.4
        )
        nan_rate = out["risk_on_score"].isna().mean() * 100
        logger.info(f"risk_on_score: NaN rate = {nan_rate:.1f}%")

    # Spot-check corr
    corr_col = "btc_SPY_corr_30d"
    if corr_col in out.columns:
        non_nan = out[corr_col].notna().sum()
        logger.info(f"{corr_col}: {non_nan}/{len(out)} non-NaN rows after daily computation")

    return out
