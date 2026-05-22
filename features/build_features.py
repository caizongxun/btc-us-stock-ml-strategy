"""Master feature assembler — combines all feature modules into one DataFrame.

Key design decisions:
- BTC uses 4h bars; stock/macro data is daily → forward-filled to 4h index.
- SHAP-based feature selection keeps top CFG.top_k_features to maintain
  a healthy sample:feature ratio (target ≥ 10:1).
"""
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


def _align_daily_to_intraday(daily_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill daily data onto an intraday index.
    Each intraday bar inherits the value of the most recent completed daily bar,
    preventing any lookahead (we only use data that was *available* at bar open).
    """
    # Reindex to union, then ffill, then select only the target bars
    combined = daily_df.reindex(daily_df.index.union(target_index))
    combined = combined.ffill()
    return combined.reindex(target_index)


def build_feature_matrix(
    target_asset: str = CFG.target_asset,
    target_days: int = CFG.target_days,
    top_k: int = CFG.top_k_features,
) -> pd.DataFrame:
    """
    Returns a fully-featured, SHAP-selected DataFrame with label columns 'y' and 'y_binary'.
    Label: +1 (long), 0 (flat), -1 (short) based on forward return over target_days bars.
    """
    logger.info("Fetching data...")
    btc = fetch_btc_ohlcv()          # 4h bars
    stocks = fetch_stocks()          # daily bars
    fg = fetch_fear_greed()          # daily sentiment

    btc_index = btc.index

    # --- BTC features (native 4h resolution) ---
    logger.info("Building BTC features...")
    df = btc.copy()
    df = add_price_features(df, prefix="btc")
    df = add_volatility_features(df, prefix="btc")
    df = add_technical_features(df, prefix="btc")
    df = detect_regime(df, prefix="btc")

    # --- Daily → 4h alignment helper ---
    def join_daily(feat_df: pd.DataFrame) -> None:
        """Align daily feature DataFrame onto BTC 4h index and left-join."""
        nonlocal df
        aligned = _align_daily_to_intraday(feat_df, btc_index)
        df = df.join(aligned, how="left")

    # --- VIX features ---
    if "^VIX" in stocks:
        vix_feats = add_vix_features(stocks["^VIX"])
        join_daily(vix_feats)

    # --- SPY features ---
    if "SPY" in stocks:
        spy = stocks["SPY"].copy()
        spy = add_price_features(spy, prefix="spy")
        spy = add_volatility_features(spy, prefix="spy")
        spy = add_technical_features(spy, prefix="spy")
        spy_feats = spy.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")
        join_daily(spy_feats)

    # --- QQQ features ---
    if "QQQ" in stocks:
        qqq = stocks["QQQ"].copy()
        qqq = add_price_features(qqq, prefix="qqq")
        qqq_feats = qqq.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")
        join_daily(qqq_feats)

    # --- Cross-asset features (daily, then align) ---
    logger.info("Building cross-asset features...")
    cross = add_cross_asset_features(btc, {k: v for k, v in stocks.items() if k != "^VIX"})
    # cross is daily-indexed (computed on daily BTC close resampled inside)
    # if cross index matches btc_index it's already 4h; otherwise align
    if not cross.index.equals(btc_index):
        cross = _align_daily_to_intraday(cross, btc_index)
    df = df.join(cross, how="left")

    # --- Fear & Greed (daily → 4h) ---
    fg_aligned = _align_daily_to_intraday(fg, btc_index)
    df = df.join(fg_aligned, how="left")
    df["fg_value"] = df["fg_value"].ffill()
    df["fg_extreme_fear"] = df["fg_extreme_fear"].ffill().fillna(0)
    df["fg_extreme_greed"] = df["fg_extreme_greed"].ffill().fillna(0)

    # --- GLD / TLT individual return features ---
    for sym in ["GLD", "TLT"]:
        if sym in stocks:
            s = stocks[sym]["close"].rename(f"{sym}_close")
            s_df = s.to_frame()
            for w in [1, 5, 10, 20]:
                s_df[f"{sym}_ret_{w}d"] = s_df[f"{sym}_close"].pct_change(w)
            s_df = s_df.drop(columns=[f"{sym}_close"])
            join_daily(s_df)

    # --- Label construction ---
    # Use native 4h close; target_days is number of 4h bars forward
    fwd_ret = btc["close"].pct_change(target_days).shift(-target_days)
    if target_asset != "BTC":
        daily_fwd = stocks[target_asset]["close"].pct_change(target_days).shift(-target_days)
        fwd_ret = _align_daily_to_intraday(daily_fwd.to_frame("fwd"), btc_index)["fwd"]

    df["fwd_ret"] = fwd_ret
    df["y"] = 0
    df.loc[fwd_ret > CFG.long_threshold, "y"] = 1
    df.loc[fwd_ret < CFG.short_threshold, "y"] = -1
    df["y_binary"] = (df["y"] == 1).astype(int)

    # Drop raw OHLCV to prevent lookahead
    raw_cols = [c for c in ["open", "high", "low", "close", "volume",
                             "taker_buy_base", "taker_buy_quote", "fwd_ret"] if c in df.columns]
    df = df.drop(columns=raw_cols)

    # Drop rows with NaN label or >50% NaN features
    df = df.dropna(subset=["y", "y_binary"])
    thresh = int(0.5 * len(df.columns))
    df = df.dropna(thresh=thresh)
    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    logger.info(f"Feature matrix (raw): {df.shape[0]} rows x {len(feature_cols)} features")

    # --- SHAP-based feature selection ---
    if top_k and top_k < len(feature_cols):
        df = _shap_feature_selection(df, feature_cols, top_k)

    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    logger.info(f"Feature matrix (final): {df.shape[0]} rows x {len(feature_cols)} features")
    return df


def _shap_feature_selection(df: pd.DataFrame, feature_cols: list, top_k: int) -> pd.DataFrame:
    """Quick SHAP importance pass with a lightweight XGBoost to select top_k features.
    Uses only the first 80% of data to avoid lookahead into the test set.
    """
    import shap
    import xgboost as xgb

    X = df[feature_cols].values
    y = df["y_binary"].values
    n_train = int(len(X) * 0.8)
    X_tr, y_tr = X[:n_train], y[:n_train]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    quick_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    quick_model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(quick_model)
    shap_vals = explainer.shap_values(X_tr)
    importance = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)

    top_features = importance.head(top_k).index.tolist()
    logger.info(f"SHAP feature selection: kept top {top_k} / {len(feature_cols)} features")
    logger.info(f"Top 10 selected:\n{importance.head(10).to_string()}")

    keep_cols = top_features + ["y", "y_binary"]
    return df[keep_cols]
