"""Cross-asset features: BTC-SPY correlation, beta, divergence, risk-on score"""
import pandas as pd
import numpy as np

def add_cross_asset_features(btc_df: pd.DataFrame, stock_dfs: dict) -> pd.DataFrame:
    """
    btc_df: BTC OHLCV DataFrame
    stock_dfs: dict of symbol -> OHLCV DataFrame
    Returns: feature DataFrame indexed same as btc_df
    """
    btc_ret = btc_df["close"].pct_change().rename("btc_ret")
    out = pd.DataFrame(index=btc_df.index)
    out["btc_ret"] = btc_ret

    for sym, sdf in stock_dfs.items():
        if sym == "^VIX":
            continue
        s_ret = sdf["close"].pct_change().rename(f"{sym}_ret")
        aligned = pd.concat([btc_ret, s_ret], axis=1, join="inner")
        s_col = f"{sym}_ret"

        # Rolling correlation
        for w in [14, 30, 60]:
            col = f"btc_{sym}_corr_{w}d"
            out[col] = aligned["btc_ret"].rolling(w).corr(aligned[s_col])

        # Rolling beta (BTC vs stock)
        for w in [30, 60]:
            cov = aligned["btc_ret"].rolling(w).cov(aligned[s_col])
            var = aligned[s_col].rolling(w).var()
            out[f"btc_{sym}_beta_{w}d"] = cov / (var + 1e-9)

        # Divergence: BTC 5d return minus stock 5d return
        btc_5d = btc_ret.rolling(5).sum()
        s_5d = s_ret.rolling(5).sum()
        out[f"btc_{sym}_div_5d"] = (btc_5d - s_5d.reindex(btc_5d.index)).fillna(0)
        out[f"btc_{sym}_div_14d"] = (
            btc_ret.rolling(14).sum() - s_ret.rolling(14).sum().reindex(btc_ret.index).fillna(0)
        )

        # Lead-lag: SPY t-1 return predicting BTC t return
        out[f"{sym}_lag1_ret"] = s_ret.shift(1).reindex(out.index)
        out[f"{sym}_lag2_ret"] = s_ret.shift(2).reindex(out.index)

        # Stock return features
        for n in [1, 3, 5, 14]:
            out[f"{sym}_ret_{n}d"] = s_ret.rolling(n).sum().reindex(out.index)

    # Risk-on composite score (higher = more risk-on)
    if "SPY" in stock_dfs and "QQQ" in stock_dfs:
        spy_ret_5d = stock_dfs["SPY"]["close"].pct_change(5).reindex(out.index)
        qqq_ret_5d = stock_dfs["QQQ"]["close"].pct_change(5).reindex(out.index)
        btc_5d_r = btc_df["close"].pct_change(5)
        out["risk_on_score"] = (
            spy_ret_5d.rank(pct=True) * 0.3
            + qqq_ret_5d.rank(pct=True) * 0.3
            + btc_5d_r.rank(pct=True) * 0.4
        )

    out = out.drop(columns=["btc_ret"])
    return out
