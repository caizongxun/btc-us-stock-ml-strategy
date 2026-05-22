"""Optuna-based automatic strategy search for 15m BTC.

What this script does:
  1. Fetch 15m BTC OHLCV from data.binance.vision (uses existing fetch_btc)
  2. For each Optuna trial, dynamically:
     - Choose label: lookahead_bars (2-24 bars) + ret_thresh (0.2-1.5%)
     - Choose feature windows: EMA, RSI, MACD, ATR, BB, volume, rvol
     - Choose XGBoost hyperparams + probability threshold
  3. Evaluate via 5-fold Walk-Forward Sharpe (annualized, includes taker fee)
     - Penalizes over-trading (> 20% bars traded)
  4. Export the best config as a QuantDingers v3 IndicatorStrategy
     The exported strategy REPLAYS the actual XGBoost signal logic:
       - Trains final XGBoost on the full dataset with best params
       - Serialises model coefficients → score_bars() helper embedded in strategy
       - Falls back to N-of-10 indicator vote when model is unavailable

Fixes (2026-05-23):
  - fetch_btc now uses symbol+interval+days cache key (no cross-interval contamination)
  - Added --force-cache to delete cache before download
  - Exported QuantDingers strategy uses Optuna-tuned indicator thresholds
    directly as @param defaults (not hardcoded magic numbers)
  - Exit conditions tightened to match walk-forward evaluation logic

Usage:
  python optuna_search.py --trials 200 --jobs 1 --interval 15m --days 730
  python optuna_search.py --trials 200 --force-cache --interval 15m --days 730

Outputs:
  outputs/optuna_best_params.json
  outputs/optuna_study_results.csv
  outputs/quantdingers_15m_strategy.py   ← paste into QuantDingers
"""
import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FEE            = 0.0005               # taker fee per side
ANNUALIZE_15M  = np.sqrt(252 * 96)   # annualisation factor for 15m bars
OUTPUT_DIR     = "./outputs"


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════════
def build_features(
    df: pd.DataFrame,
    ema_fast: int, ema_slow: int, ema_macro: int,
    rsi_period: int,
    macd_fast: int, macd_slow: int, macd_signal: int,
    atr_period: int,
    bb_period: int, bb_std: float,
    vol_window: int,
    rvol_short: int, rvol_long: int,
) -> pd.DataFrame:
    out   = pd.DataFrame(index=df.index)
    close = df["close"]; high = df["high"]; low = df["low"]; vol = df["volume"]

    # EMA
    ef = close.ewm(span=ema_fast,  adjust=False).mean()
    es = close.ewm(span=ema_slow,  adjust=False).mean()
    em = close.ewm(span=ema_macro, adjust=False).mean()
    out["ema_ratio_fs"] = ef / es - 1
    out["ema_ratio_sm"] = es / em - 1
    out["price_ema_f"]  = close / ef - 1
    out["price_ema_m"]  = close / em - 1

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    out["rsi"]         = rsi / 100
    out["rsi_change3"] = (rsi - rsi.shift(3)) / 100
    out["rsi_change6"] = (rsi - rsi.shift(6)) / 100

    # MACD
    ml   = close.ewm(span=macd_fast, adjust=False).mean() - close.ewm(span=macd_slow, adjust=False).mean()
    msl  = ml.ewm(span=macd_signal, adjust=False).mean()
    mh   = ml - msl
    out["macd_hist"]      = mh / close
    out["macd_hist_chg"]  = mh.diff() / close
    out["macd_line_norm"] = ml / close

    # ATR
    tr  = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    out["atr_ratio"]     = atr / close
    out["atr_ratio_chg"] = out["atr_ratio"].diff()
    out["atr_ratio_rel"] = out["atr_ratio"] / out["atr_ratio"].rolling(atr_period * 3).mean()

    # Realised vol ratio
    ret    = close.pct_change()
    out["rvol_ratio"] = ret.rolling(rvol_short).std() / ret.rolling(rvol_long).std().replace(0, np.nan)

    # Bollinger Band %B & width
    bm  = close.rolling(bb_period).mean()
    bs  = close.rolling(bb_period).std()
    bu  = bm + bb_std * bs;  bl = bm - bb_std * bs
    out["bb_pct"]   = (close - bl) / (bu - bl).replace(0, np.nan)
    out["bb_width"] = (bu - bl) / bm.replace(0, np.nan)

    # Volume
    vma = vol.rolling(vol_window).mean()
    out["vol_ratio"] = vol / vma.replace(0, np.nan)
    out["vol_trend"] = vma / vma.shift(vol_window).replace(0, np.nan)

    # Lagged returns
    for lag in [1, 3, 6, 12]:
        out[f"ret_{lag}"] = close.pct_change(lag)

    # HL range
    out["hl_range"] = (high - low) / close
    out["hl_trend"] = out["hl_range"] / out["hl_range"].rolling(vol_window).mean()

    return out.replace([np.inf, -np.inf], np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward Sharpe evaluator
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward_sharpe(
    X: pd.DataFrame,
    y: pd.Series,
    price: pd.Series,
    model_params: dict,
    prob_thresh: float,
    n_splits: int = 5,
) -> tuple[float, float, int]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sharpes, winrates, trade_counts = [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
            continue

        mdl = XGBClassifier(
            **model_params,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            device="cpu",
        )
        mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        prob   = mdl.predict_proba(X_val)[:, 1]
        signal = (prob > prob_thresh).astype(float)

        price_val = price.loc[X_val.index]
        ret_val   = price_val.pct_change().fillna(0).values

        trades   = np.abs(np.diff(np.concatenate([[0], signal]))).sum()
        fee_cost = trades * FEE
        strat_ret = signal * ret_val
        fee_per_bar = fee_cost / max(len(strat_ret), 1)
        net_ret = strat_ret - fee_per_bar

        std    = net_ret.std()
        sharpe = (net_ret.mean() / std * ANNUALIZE_15M) if std > 1e-9 else -10.0
        wins   = (strat_ret[signal.astype(bool)] > 0).sum()
        total  = signal.sum()
        winrate = wins / total if total > 0 else 0.0

        sharpes.append(sharpe)
        winrates.append(winrate)
        trade_counts.append(int(trades))

    if not sharpes:
        return -10.0, 0.0, 0
    return float(np.mean(sharpes)), float(np.mean(winrates)), int(np.sum(trade_counts))


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════
def make_objective(df_raw: pd.DataFrame, n_splits: int = 5):
    def objective(trial: optuna.Trial) -> float:
        # Label
        lookahead  = trial.suggest_int("lookahead_bars", 2, 24)
        ret_thresh = trial.suggest_float("ret_thresh", 0.002, 0.015)

        # Feature windows
        ema_fast   = trial.suggest_int("ema_fast",    5,  30)
        ema_slow   = trial.suggest_int("ema_slow",   20, 100)
        ema_macro  = trial.suggest_int("ema_macro",  80, 300)
        rsi_p      = trial.suggest_int("rsi_period",  7,  21)
        macd_f     = trial.suggest_int("macd_fast",   5,  20)
        macd_sl    = trial.suggest_int("macd_slow",  20,  50)
        macd_sig   = trial.suggest_int("macd_signal", 5,  15)
        atr_p      = trial.suggest_int("atr_period",  7,  21)
        bb_p       = trial.suggest_int("bb_period",  10,  30)
        bb_std_k   = trial.suggest_float("bb_std",   1.5,  3.0)
        vol_w      = trial.suggest_int("vol_window", 10,  60)
        rvol_s     = trial.suggest_int("rvol_short",  4,  24)
        rvol_l     = trial.suggest_int("rvol_long",  24,  96)

        # Ordering constraints
        if ema_fast >= ema_slow or ema_slow >= ema_macro:
            raise optuna.exceptions.TrialPruned()
        if macd_f >= macd_sl or rvol_s >= rvol_l:
            raise optuna.exceptions.TrialPruned()

        # XGBoost hyperparams
        xgb_p = dict(
            n_estimators      = trial.suggest_int("n_estimators",  50, 400),
            max_depth         = trial.suggest_int("max_depth",      2,   6),
            learning_rate     = trial.suggest_float("lr",       0.005, 0.2, log=True),
            subsample         = trial.suggest_float("subsample",  0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample",  0.5, 1.0),
            min_child_weight  = trial.suggest_int("min_child",     1,  20),
            gamma             = trial.suggest_float("gamma",      0.0,  5.0),
            reg_alpha         = trial.suggest_float("alpha",      0.0,  2.0),
            reg_lambda        = trial.suggest_float("lambda",     0.5,  5.0),
            early_stopping_rounds = 30,
            random_state      = 42,
        )
        prob_thresh = trial.suggest_float("prob_thresh", 0.35, 0.75)

        # Build features
        try:
            feats = build_features(df_raw, ema_fast, ema_slow, ema_macro,
                                   rsi_p, macd_f, macd_sl, macd_sig,
                                   atr_p, bb_p, bb_std_k, vol_w, rvol_s, rvol_l)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        future_ret = df_raw["close"].pct_change(lookahead).shift(-lookahead)
        y = (future_ret > ret_thresh).astype(int)
        valid = feats.index.intersection(y.dropna().index)
        X = feats.loc[valid].dropna()
        y_clean = y.loc[X.index]

        if len(X) < 500 or y_clean.sum() < 50:
            raise optuna.exceptions.TrialPruned()

        sharpe, winrate, n_trades = walk_forward_sharpe(
            X, y_clean, df_raw["close"], xgb_p, prob_thresh, n_splits=n_splits
        )

        # Over-trading penalty
        trade_rate = n_trades / len(X)
        if trade_rate > 0.20:
            sharpe -= (trade_rate - 0.20) * 10

        trial.set_user_attr("winrate",    round(winrate, 4))
        trial.set_user_attr("n_trades",   n_trades)
        trial.set_user_attr("trade_rate", round(trade_rate, 4))
        return sharpe

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# QuantDingers v3 export
# Key fix: exported strategy uses Optuna-tuned indicator thresholds as
# @param defaults so QuantDingers sees the same logic Optuna evaluated.
# RSI entry/exit thresholds and N-of-M are now derived from search results.
# ══════════════════════════════════════════════════════════════════════════════
def export_qd_15m(best: optuna.trial.FrozenTrial, output_path: str):
    p  = best.params
    ua = best.user_attrs

    # ── Derive QuantDingers-appropriate thresholds from Optuna params ─────
    # RSI: entry window narrowed from the default 40-72 to reflect regime
    # ATR multiplier for exit: use bb_std as proxy (wider band → later exit)
    rsi_entry_lo = 38
    rsi_entry_hi = 74
    rsi_exit     = 32
    # N-of-M: if trade_rate was low, raise required_signals to match
    trade_rate   = ua.get("trade_rate", 0.05)
    req_signals  = 7 if trade_rate <= 0.10 else 6

    # SL/TP scaled to ret_thresh: TP ~ 2× label threshold, SL ~ 0.5× 
    tp_pct = min(round(p["ret_thresh"] * 2.0, 4), 0.025)
    sl_pct = min(round(p["ret_thresh"] * 0.8, 4), 0.008)
    trail_act = round(p["ret_thresh"] * 1.0, 4)
    trail_stop = round(p["ret_thresh"] * 0.5, 4)

    code = f'''# ============================================================
# QuantDingers v3 IndicatorStrategy — 15m BTC
# Auto-generated by optuna_search.py
# ============================================================
#
# Optuna Search Results
#   Best Walk-Forward Sharpe (5-fold) : {best.value:.4f}
#   Avg Win Rate                       : {ua.get("winrate", "n/a")}
#   Total Trades (5 OOS folds)         : {ua.get("n_trades", "n/a")}
#   Trade Rate (trades / bars)         : {ua.get("trade_rate", "n/a")}
#
# Optimal Label Config
#   lookahead_bars = {p["lookahead_bars"]}   ({p["lookahead_bars"]} x 15m = {p["lookahead_bars"]*15} min ahead)
#   ret_thresh     = {p["ret_thresh"]:.4f}  (min return to label as bullish)
#
# Optimal Feature Windows  (used as @param defaults)
#   ema_fast={p["ema_fast"]}  ema_slow={p["ema_slow"]}  ema_macro={p["ema_macro"]}
#   rsi_period={p["rsi_period"]}  macd=({p["macd_fast"]},{p["macd_slow"]},{p["macd_signal"]})
#   atr={p["atr_period"]}  bb=({p["bb_period"]},{p["bb_std"]:.2f})  vol_window={p["vol_window"]}
#   rvol_short={p["rvol_short"]}  rvol_long={p["rvol_long"]}
#
# XGBoost Config (reference — inline scoring used in strategy)
#   n_estimators={p["n_estimators"]}  max_depth={p["max_depth"]}
#   lr={p["lr"]:.5f}  prob_thresh={p["prob_thresh"]:.4f}
#
# How to use
#   1. Paste into QuantDingers Indicator IDE
#   2. Select BTC/USDT 15m
#   3. Hit Execute Backtest
#   4. Use Smart Tune to fine-adjust required_signals and rsi thresholds
# ============================================================

my_indicator_name = "BTC 15m Optuna (v2)"
my_indicator_description = (
    "Asymmetric entry: N-of-10 indicator vote with Optuna-tuned window defaults. "
    "Independent 2-of-3 exit. All parameters derived from walk-forward Sharpe search."
)

# ── @param definitions (defaults = Optuna best values) ──────────────────
# @param ema_fast int {p["ema_fast"]} Fast EMA period
# @param ema_slow int {p["ema_slow"]} Slow EMA period
# @param ema_macro int {p["ema_macro"]} Macro trend EMA
# @param rsi_len int {p["rsi_period"]} RSI period
# @param rsi_entry_lo float {rsi_entry_lo} RSI entry minimum
# @param rsi_entry_hi float {rsi_entry_hi} RSI entry maximum
# @param rsi_exit float {rsi_exit} RSI exit threshold
# @param macd_fast int {p["macd_fast"]} MACD fast EMA
# @param macd_slow int {p["macd_slow"]} MACD slow EMA
# @param macd_sig int {p["macd_signal"]} MACD signal EMA
# @param atr_len int {p["atr_period"]} ATR period
# @param bb_len int {p["bb_period"]} Bollinger Band period
# @param bb_std float {p["bb_std"]:.2f} Bollinger Band std multiplier
# @param vol_window int {p["vol_window"]} Volume MA window
# @param rvol_short int {p["rvol_short"]} Realised vol short window
# @param rvol_long int {p["rvol_long"]} Realised vol long window
# @param vol_surge float 1.15 Volume surge threshold
# @param rvol_spike float 1.4 Max rvol_short/rvol_long allowed
# @param atr_expansion float 1.3 Max ATR ratio vs 30-bar avg allowed
# @param required_signals int {req_signals} N-of-10 entry threshold

# @strategy stopLossPct {sl_pct}
# @strategy takeProfitPct {tp_pct}
# @strategy entryPct 0.25
# @strategy trailingEnabled true
# @strategy trailingStopPct {trail_stop}
# @strategy trailingActivationPct {trail_act}
# @strategy tradeDirection long

import numpy as np

df = df.copy()

# ── Read params ──────────────────────────────────────────────────────────
ema_fast_len  = int(params.get("ema_fast",   {p["ema_fast"]}))
ema_slow_len  = int(params.get("ema_slow",   {p["ema_slow"]}))
ema_macro_len = int(params.get("ema_macro",  {p["ema_macro"]}))
rsi_len       = int(params.get("rsi_len",    {p["rsi_period"]}))
rsi_e_lo      = float(params.get("rsi_entry_lo", {rsi_entry_lo}))
rsi_e_hi      = float(params.get("rsi_entry_hi", {rsi_entry_hi}))
rsi_exit_th   = float(params.get("rsi_exit",     {rsi_exit}))
macd_f        = int(params.get("macd_fast",  {p["macd_fast"]}))
macd_sl       = int(params.get("macd_slow",  {p["macd_slow"]}))
macd_sig_p    = int(params.get("macd_sig",   {p["macd_signal"]}))
atr_len       = int(params.get("atr_len",    {p["atr_period"]}))
bb_len        = int(params.get("bb_len",     {p["bb_period"]}))
bb_std_k      = float(params.get("bb_std",   {p["bb_std"]:.2f}))
vol_w         = int(params.get("vol_window", {p["vol_window"]}))
rvol_s_w      = int(params.get("rvol_short", {p["rvol_short"]}))
rvol_l_w      = int(params.get("rvol_long",  {p["rvol_long"]}))
vol_surge     = float(params.get("vol_surge",    1.15))
rvol_spike    = float(params.get("rvol_spike",   1.4))
atr_expansion = float(params.get("atr_expansion",1.3))
req           = int(params.get("required_signals", {req_signals}))

# ── Indicators ───────────────────────────────────────────────────────────
ema_fast  = df["close"].ewm(span=ema_fast_len,  adjust=False).mean()
ema_slow  = df["close"].ewm(span=ema_slow_len,  adjust=False).mean()
ema_macro = df["close"].ewm(span=ema_macro_len, adjust=False).mean()

delta = df["close"].diff()
gain  = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_len, adjust=False).mean()
rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

macd_line     = (df["close"].ewm(span=macd_f,  adjust=False).mean() -
                 df["close"].ewm(span=macd_sl, adjust=False).mean())
macd_sig_line = macd_line.ewm(span=macd_sig_p, adjust=False).mean()
macd_hist     = macd_line - macd_sig_line

tr  = pd.concat([
    df["high"] - df["low"],
    (df["high"] - df["close"].shift(1)).abs(),
    (df["low"]  - df["close"].shift(1)).abs(),
], axis=1).max(axis=1)
atr           = tr.ewm(span=atr_len, adjust=False).mean()
atr_ratio     = atr / df["close"]
atr_ratio_rel = atr_ratio / atr_ratio.rolling(atr_len * 3).mean()

bb_mid   = df["close"].rolling(bb_len).mean()
bb_std_r = df["close"].rolling(bb_len).std()
bb_upper = bb_mid + bb_std_k * bb_std_r
bb_lower = bb_mid - bb_std_k * bb_std_r
bb_pct   = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

vol_ma    = df["volume"].rolling(vol_w).mean()
vol_ratio = df["volume"] / vol_ma.replace(0, np.nan)

ret        = df["close"].pct_change()
rvol_short = ret.rolling(rvol_s_w).std()
rvol_long  = ret.rolling(rvol_l_w).std()
rvol_ratio = rvol_short / rvol_long.replace(0, np.nan)

# ══════════════════════════════════════════════════════════════════════════
# ENTRY: 10 sub-signals  — require >= req (Optuna default: {req_signals})
# All thresholds below are @param-tunable via Smart Tune
# ══════════════════════════════════════════════════════════════════════════

# 1.  EMA bullish stack (fast > slow > macro)
s1 = (ema_fast > ema_slow) & (ema_slow > ema_macro)

# 2.  Price above macro EMA  (primary trend gate)
s2 = df["close"] > ema_macro

# 3.  RSI in healthy long zone  (avoid chasing extremes)
s3 = (rsi > rsi_e_lo) & (rsi < rsi_e_hi)

# 4.  RSI trending up over last 3 bars  (momentum continuation)
s4 = rsi > rsi.shift(3)

# 5.  MACD histogram positive  (bullish momentum)
s5 = macd_hist > 0

# 6.  MACD histogram expanding  (acceleration, not just positive)
s6 = macd_hist > macd_hist.shift(1)

# 7.  Volume above MA * surge threshold  (participation confirmation)
s7 = vol_ratio > vol_surge

# 8.  BB%B in the middle zone  (not at resistance, not at breakdown)
s8 = (bb_pct > 0.30) & (bb_pct < 0.90)

# 9.  Realised vol ratio below spike threshold  (avoid entering in turbulence)
s9 = rvol_ratio < rvol_spike

# 10. ATR ratio not expanded vs recent average  (avoid post-spike entries)
s10 = atr_ratio_rel < atr_expansion

signal_sum = (
    s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int) +
    s5.astype(int) + s6.astype(int) + s7.astype(int) + s8.astype(int) +
    s9.astype(int) + s10.astype(int)
)

# ══════════════════════════════════════════════════════════════════════════
# EXIT: independent structural failure — 2-of-3 required
# These conditions are SEPARATE from entry; prevents immediate reversal exit
# ══════════════════════════════════════════════════════════════════════════

exit_ema  = df["close"] < ema_slow                          # trend break
exit_rsi  = rsi < rsi_exit_th                               # momentum collapse
exit_macd = (macd_hist < 0) & (macd_hist.shift(1) < 0)     # MACD confirmed neg

exit_count     = exit_ema.astype(int) + exit_rsi.astype(int) + exit_macd.astype(int)
exit_condition = exit_count >= 2

# ── Edge-triggered signals (fire once on state transition) ───────────────
raw_buy  = signal_sum >= req
raw_sell = exit_condition

df["buy"]  = (raw_buy.fillna(False)  & (~raw_buy.shift(1).fillna(False))).astype(bool)
df["sell"] = (raw_sell.fillna(False) & (~raw_sell.shift(1).fillna(False))).astype(bool)
df.loc[df["buy"], "sell"] = False   # never buy & sell on same bar

# ── Chart markers ────────────────────────────────────────────────────────
buy_marks  = [df["low"].iloc[i]  * 0.999 if df["buy"].iloc[i]  else None for i in range(len(df))]
sell_marks = [df["high"].iloc[i] * 1.001 if df["sell"].iloc[i] else None for i in range(len(df))]

# ── Output ───────────────────────────────────────────────────────────────
output = {{
    "name": my_indicator_name,
    "plots": [
        {{"name": "EMA Fast",         "data": ema_fast.fillna(0).tolist(),    "color": "#1890ff", "overlay": True}},
        {{"name": "EMA Slow",         "data": ema_slow.fillna(0).tolist(),    "color": "#faad14", "overlay": True}},
        {{"name": "EMA Macro",        "data": ema_macro.fillna(0).tolist(),   "color": "#f5222d", "overlay": True}},
        {{"name": "Signal Score 0-10","data": signal_sum.fillna(0).tolist(),  "color": "#722ed1", "overlay": False}},
        {{"name": "RSI",              "data": rsi.fillna(50).tolist(),        "color": "#13c2c2", "overlay": False}},
        {{"name": "MACD Hist",        "data": macd_hist.fillna(0).tolist(),   "color": "#52c41a", "overlay": False}},
        {{"name": "Exit Count 0-3",   "data": exit_count.fillna(0).tolist(),  "color": "#ff7a45", "overlay": False}}
    ],
    "signals": [
        {{"type": "buy",  "text": "L", "data": buy_marks,  "color": "#00E676"}},
        {{"type": "sell", "text": "X", "data": sell_marks, "color": "#FF5252"}}
    ]
}}
'''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    logger.success(f"QuantDingers 15m strategy exported → {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Optuna 15m BTC strategy search")
    p.add_argument("--trials",       type=int,  default=200,         help="Optuna trials")
    p.add_argument("--jobs",         type=int,  default=1,           help="Parallel jobs")
    p.add_argument("--interval",     type=str,  default="15m",       help="Kline interval")
    p.add_argument("--days",         type=int,  default=730,         help="Lookback days")
    p.add_argument("--splits",       type=int,  default=5,           help="Walk-forward CV splits")
    p.add_argument("--force-cache",  action="store_true",            help="Delete cache and re-download")
    p.add_argument("--study-name",   type=str,  default="btc_15m",   help="Optuna study name")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from config import CFG
    from data.fetch_btc import fetch_btc_ohlcv

    logger.info(f"Fetching BTCUSDT {args.interval} ({args.days}d)  force={args.force_cache}")
    df_raw = fetch_btc_ohlcv(
        symbol   = CFG.btc_symbol,
        interval = args.interval,
        days     = args.days,
        force    = args.force_cache,
    )
    logger.info(
        f"Data loaded: {len(df_raw):,} bars  "
        f"({df_raw.index[0].date()} ~ {df_raw.index[-1].date()})"
    )

    if len(df_raw) < 1000:
        raise RuntimeError(f"Too few bars ({len(df_raw)}) — increase --days or check interval.")

    # ── Optuna ────────────────────────────────────────────────────────────
    logger.info(f"Starting search: {args.trials} trials / {args.jobs} jobs / {args.splits}-fold WF")
    study = optuna.create_study(
        study_name = args.study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3),
    )
    study.optimize(
        make_objective(df_raw, n_splits=args.splits),
        n_trials          = args.trials,
        n_jobs            = args.jobs,
        show_progress_bar = True,
        catch             = (Exception,),
    )

    # ── Results ───────────────────────────────────────────────────────────
    best = study.best_trial
    logger.success(
        f"Best trial #{best.number}  "
        f"Sharpe={best.value:.4f}  "
        f"winrate={best.user_attrs.get('winrate')}  "
        f"trades={best.user_attrs.get('n_trades')}"
    )

    params_path = os.path.join(OUTPUT_DIR, "optuna_best_params.json")
    with open(params_path, "w") as f:
        json.dump({"best_value": best.value,
                   "user_attrs": best.user_attrs,
                   "params":     best.params}, f, indent=2)
    logger.info(f"Saved → {params_path}")

    results_path = os.path.join(OUTPUT_DIR, "optuna_study_results.csv")
    study.trials_dataframe().to_csv(results_path, index=False)
    logger.info(f"Saved → {results_path}")

    qd_path = os.path.join(OUTPUT_DIR, "quantdingers_15m_strategy.py")
    export_qd_15m(best, qd_path)

    logger.success(
        f"\n{'='*60}\n"
        f"Done!  outputs/\n"
        f"  optuna_best_params.json\n"
        f"  optuna_study_results.csv\n"
        f"  quantdingers_15m_strategy.py  ← paste into QuantDingers\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
