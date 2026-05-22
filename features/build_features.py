"""Master feature assembler.

Key design decisions:
- BTC uses 4h bars; stock/macro data is daily.
- Cross-asset features (corr, beta, risk_on_score) are computed on BTC
  **daily-resampled** data so that rolling windows (30d, 60d) cover real
  calendar days, not 4h-bar counts.
- After computation, cross-asset features are forward-filled back to 4h index.
- SHAP-based feature selection keeps top CFG.top_k_features.
- SIGNAL_COLS are always preserved after SHAP selection.
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

SIGNAL_COLS = [
    "btc_trend_score",
    "btc_regime_bear",
    "btc_regime_sideways",
    "btc_rsi14_ob",
    "btc_vol_ma_ratio_14d",
    "btc_vol_ma_ratio_7d",
    "btc_SPY_corr_30d",
    "fg_value",
    "risk_on_score",
    "vix_low",
]


def _resample_btc_daily(btc_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4h BTC OHLCV to daily for cross-asset feature computation."""
    daily = btc_4h.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return daily


def _align_daily_to_intraday(
    daily_df: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Forward-fill daily data onto an intraday (4h) index."""
    combined = daily_df.reindex(daily_df.index.union(target_index))
    combined = combined.ffill()
    return combined.reindex(target_index)


def build_feature_matrix(
    target_asset: str = CFG.target_asset,
    target_days: int = CFG.target_days,
    top_k: int = CFG.top_k_features,
) -> pd.DataFrame:
    logger.info("Fetching data...")
    btc = fetch_btc_ohlcv()          # 4h index
    stocks = fetch_stocks()           # daily index
    fg = fetch_fear_greed()           # daily index

    btc_index = btc.index
    btc_daily = _resample_btc_daily(btc)   # daily — used for cross-asset only

    # ------------------------------------------------------------------ #
    # BTC 4h features
    # ------------------------------------------------------------------ #
    logger.info("Building BTC 4h features...")
    df = btc.copy()
    df = add_price_features(df, prefix="btc")
    df = add_volatility_features(df, prefix="btc")
    df = add_technical_features(df, prefix="btc")
    df = detect_regime(df, prefix="btc")

    def join_daily(feat_df: pd.DataFrame) -> None:
        nonlocal df
        overlap = [c for c in feat_df.columns if c in df.columns]
        if overlap:
            feat_df = feat_df.drop(columns=overlap)
        if feat_df.empty:
            return
        aligned = _align_daily_to_intraday(feat_df, btc_index)
        df = df.join(aligned, how="left")

    # ------------------------------------------------------------------ #
    # Stock daily features (aligned to 4h)
    # ------------------------------------------------------------------ #
    if "^VIX" in stocks:
        vix_feats = add_vix_features(stocks["^VIX"])
        join_daily(vix_feats)

    if "SPY" in stocks:
        spy = stocks["SPY"].copy()
        spy = add_price_features(spy, prefix="spy")
        spy = add_volatility_features(spy, prefix="spy")
        spy = add_technical_features(spy, prefix="spy")
        spy_feats = spy.drop(
            columns=["open", "high", "low", "close", "volume"], errors="ignore"
        )
        join_daily(spy_feats)

    if "QQQ" in stocks:
        qqq = stocks["QQQ"].copy()
        qqq = add_price_features(qqq, prefix="qqq")
        qqq_feats = qqq.drop(
            columns=["open", "high", "low", "close", "volume"], errors="ignore"
        )
        join_daily(qqq_feats)

    # ------------------------------------------------------------------ #
    # Cross-asset features — computed on DAILY BTC to match stock index
    # ------------------------------------------------------------------ #
    logger.info("Building cross-asset features on daily BTC...")
    stocks_no_vix = {k: v for k, v in stocks.items() if k != "^VIX"}
    cross = add_cross_asset_features(btc_daily, stocks_no_vix)

    # cross is on daily index — forward-fill to 4h
    logger.info(f"Cross-asset features: {cross.shape[1]} cols, {cross.notna().mean().describe()}")
    join_daily(cross)

    # Spot-check: log NaN rate for key signal cols before fillna
    for col in ["btc_SPY_corr_30d", "risk_on_score"]:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            logger.info(f"  {col}: NaN rate before fillna = {nan_pct:.1f}%")

    # ------------------------------------------------------------------ #
    # Fear & Greed
    # ------------------------------------------------------------------ #
    fg_aligned = _align_daily_to_intraday(fg, btc_index)
    df = df.join(fg_aligned, how="left")
    df["fg_value"] = df["fg_value"].ffill()
    df["fg_extreme_fear"] = df["fg_extreme_fear"].ffill().fillna(0)
    df["fg_extreme_greed"] = df["fg_extreme_greed"].ffill().fillna(0)

    # ------------------------------------------------------------------ #
    # Labels
    # ------------------------------------------------------------------ #
    fwd_ret = btc["close"].pct_change(target_days).shift(-target_days)
    if target_asset != "BTC":
        daily_fwd = (
            stocks[target_asset]["close"].pct_change(target_days).shift(-target_days)
        )
        fwd_ret = _align_daily_to_intraday(daily_fwd.to_frame("fwd"), btc_index)["fwd"]

    df["fwd_ret"] = fwd_ret
    df["y"] = 0
    df.loc[fwd_ret > CFG.long_threshold, "y"] = 1
    df.loc[fwd_ret < -CFG.short_threshold, "y"] = -1
    df["y_binary"] = (df["y"] == 1).astype(int)

    # Drop raw OHLCV
    raw_cols = [
        c for c in ["open", "high", "low", "close", "volume",
                    "taker_buy_base", "taker_buy_quote", "fwd_ret"]
        if c in df.columns
    ]
    df = df.drop(columns=raw_cols)

    df = df.dropna(subset=["y", "y_binary"])
    thresh = int(0.5 * len(df.columns))
    df = df.dropna(thresh=thresh)
    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    logger.info(f"Feature matrix (raw): {df.shape[0]} rows x {len(feature_cols)} features")

    if top_k and top_k < len(feature_cols):
        df = _shap_feature_selection(df, feature_cols, top_k)

    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    logger.info(f"Feature matrix (final): {df.shape[0]} rows x {len(feature_cols)} features")
    return df


def _shap_feature_selection(
    df: pd.DataFrame, feature_cols: list, top_k: int
) -> pd.DataFrame:
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

    present_signal_cols = [c for c in SIGNAL_COLS if c in feature_cols]
    top_features = importance.head(top_k).index.tolist()
    all_keep = list(dict.fromkeys(top_features + present_signal_cols))

    n_added = len([c for c in present_signal_cols if c not in top_features])
    logger.info(
        f"SHAP selection: top {top_k} + {n_added} forced signal cols = {len(all_keep)} total"
    )

    keep_cols = all_keep + ["y", "y_binary"]
    return df[keep_cols]
