"""Master feature assembler — combines all feature modules into one DataFrame"""
import pandas as pd
import numpy as np
from loguru import logger

from data.fetch_btc import fetch_btc_ohlcv
from data.fetch_stocks import fetch_stocks
from data.fetch_onchain import fetch_fear_greed
from features.price_features import add_price_features
from features.volatility_features import add_volatility_features, add_vix_features
from features.cross_asset_features import add_cross_asset_features
from features.technical_features import add_technical_features
from features.regime_features import detect_regime
from config import CFG

def build_feature_matrix(target_asset: str = CFG.target_asset, target_days: int = CFG.target_days) -> pd.DataFrame:
    """
    Returns a fully-featured DataFrame with label column 'y'.
    Label: +1 (long), 0 (flat), -1 (short) based on forward return.
    """
    logger.info("Fetching data...")
    btc = fetch_btc_ohlcv()
    stocks = fetch_stocks()
    fg = fetch_fear_greed()

    # --- BTC features ---
    logger.info("Building BTC features...")
    df = btc.copy()
    df = add_price_features(df, prefix="btc")
    df = add_volatility_features(df, prefix="btc")
    df = add_technical_features(df, prefix="btc")
    df = detect_regime(df, prefix="btc")

    # --- VIX features ---
    if "^VIX" in stocks:
        vix_feats = add_vix_features(stocks["^VIX"])
        df = df.join(vix_feats, how="left")

    # --- SPY features ---
    if "SPY" in stocks:
        spy = stocks["SPY"].copy()
        spy = add_price_features(spy, prefix="spy")
        spy = add_volatility_features(spy, prefix="spy")
        spy = add_technical_features(spy, prefix="spy")
        df = df.join(spy.drop(columns=["open","high","low","close","volume"], errors="ignore"),
                     how="left")

    # --- QQQ features ---
    if "QQQ" in stocks:
        qqq = stocks["QQQ"].copy()
        qqq = add_price_features(qqq, prefix="qqq")
        df = df.join(qqq.drop(columns=["open","high","low","close","volume"], errors="ignore"),
                     how="left")

    # --- Cross-asset features ---
    logger.info("Building cross-asset features...")
    cross = add_cross_asset_features(btc, {k: v for k, v in stocks.items() if k != "^VIX"})
    df = df.join(cross, how="left")

    # --- Fear & Greed ---
    df = df.join(fg, how="left")
    df["fg_value"] = df["fg_value"].ffill()
    df["fg_extreme_fear"] = df["fg_extreme_fear"].ffill().fillna(0)
    df["fg_extreme_greed"] = df["fg_extreme_greed"].ffill().fillna(0)

    # --- Label construction ---
    if target_asset == "BTC":
        fwd_ret = btc["close"].pct_change(target_days).shift(-target_days)
    else:
        fwd_ret = stocks[target_asset]["close"].pct_change(target_days).shift(-target_days)
        fwd_ret = fwd_ret.reindex(df.index)

    df["fwd_ret"] = fwd_ret
    df["y"] = 0
    df.loc[fwd_ret > CFG.long_threshold, "y"] = 1
    df.loc[fwd_ret < CFG.short_threshold, "y"] = -1

    # --- Binary label for classifier ---
    df["y_binary"] = (df["y"] == 1).astype(int)  # 1=long, 0=else

    # Drop OHLCV raw columns to prevent lookahead
    raw_cols = [c for c in ["open","high","low","close","volume",
                              "taker_buy_base","taker_buy_quote","fwd_ret"] if c in df.columns]
    df = df.drop(columns=raw_cols)

    # Drop rows with NaN label or >50% NaN features
    df = df.dropna(subset=["y", "y_binary"])
    thresh = int(0.5 * len(df.columns))
    df = df.dropna(thresh=thresh)
    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    logger.info(f"Feature matrix: {df.shape[0]} rows x {len(feature_cols)} features")
    return df
