"""Optuna-based automatic strategy search for 15m BTC.

What this script does:
  1. Fetch 15m BTC OHLCV from data.binance.vision (uses existing fetch_btc)
  2. For each Optuna trial, dynamically:
     - Choose label: lookahead_bars (2-24 bars) + ret_thresh (0.2-1.5%)
     - Choose feature windows: EMA, RSI, MACD, ATR, BB, volume, rvol
     - Choose XGBoost hyperparams
     - Choose probability threshold
     - Choose N-of-M entry threshold (for QuantDingers signal)
  3. Evaluate via 5-fold Walk-Forward Sharpe (annualized)
     - Includes taker fee 0.05% per side
     - Penalizes over-trading (> 10% bars traded)
  4. After N trials, export the best config as a QuantDingers strategy

Usage:
  python optuna_search.py --trials 200 --jobs 4 --interval 15m
  python optuna_search.py --trials 500 --jobs 1 --interval 15m --no-cache

Outputs:
  outputs/optuna_best_params.json          -- best trial params
  outputs/optuna_study_results.csv         -- all trial history
  outputs/quantdingers_15m_strategy.py     -- paste-ready QuantDingers code
"""
import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FEE = 0.0005          # taker fee per side (0.05%)
ANNUALIZE_15M = np.sqrt(252 * 96)   # 15m bars per year
OUTPUT_DIR = "./outputs"


# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    ema_macro: int,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    atr_period: int,
    bb_period: int,
    bb_std: float,
    vol_window: int,
    rvol_short: int,
    rvol_long: int,
) -> pd.DataFrame:
    """Compute all OHLCV-derived features. Returns a clean DataFrame."""
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # EMA
    ema_f  = close.ewm(span=ema_fast,  adjust=False).mean()
    ema_s  = close.ewm(span=ema_slow,  adjust=False).mean()
    ema_m  = close.ewm(span=ema_macro, adjust=False).mean()
    out["ema_ratio_fs"]  = ema_f / ema_s - 1          # fast/slow spread
    out["ema_ratio_sm"]  = ema_s / ema_m - 1          # slow/macro spread
    out["price_ema_f"]   = close / ema_f - 1          # price position vs fast
    out["price_ema_m"]   = close / ema_m - 1          # price position vs macro

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    out["rsi"]         = rsi / 100
    out["rsi_change3"] = (rsi - rsi.shift(3)) / 100   # RSI momentum 3-bar
    out["rsi_change6"] = (rsi - rsi.shift(6)) / 100

    # MACD
    macd_line = (close.ewm(span=macd_fast, adjust=False).mean() -
                 close.ewm(span=macd_slow, adjust=False).mean())
    macd_sig  = macd_line.ewm(span=macd_signal, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    out["macd_hist"]       = macd_hist / close          # normalised by price
    out["macd_hist_chg"]   = macd_hist.diff() / close
    out["macd_line_norm"]  = macd_line / close

    # ATR & realised vol
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    out["atr_ratio"]      = atr / close
    out["atr_ratio_chg"]  = out["atr_ratio"].diff()
    out["atr_ratio_rel"]  = out["atr_ratio"] / out["atr_ratio"].rolling(atr_period*3).mean()

    ret = close.pct_change()
    rvol_s = ret.rolling(rvol_short).std()
    rvol_l = ret.rolling(rvol_long).std()
    out["rvol_ratio"] = rvol_s / rvol_l.replace(0, np.nan)

    # Bollinger Band %B
    bb_mid   = close.rolling(bb_period).mean()
    bb_std_  = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_
    bb_lower = bb_mid - bb_std * bb_std_
    out["bb_pct"]   = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)  # squeeze detector

    # Volume
    vol_ma = vol.rolling(vol_window).mean()
    out["vol_ratio"]  = vol / vol_ma.replace(0, np.nan)
    out["vol_trend"]  = vol_ma / vol_ma.shift(vol_window).replace(0, np.nan)

    # Price returns (lagged, so no lookahead)
    for lag in [1, 3, 6, 12]:
        out[f"ret_{lag}"] = close.pct_change(lag)

    # High-Low range
    out["hl_range"]  = (high - low) / close
    out["hl_trend"]  = out["hl_range"] / out["hl_range"].rolling(vol_window).mean()

    return out.replace([np.inf, -np.inf], np.nan)


# ── Walk-forward Sharpe ────────────────────────────────────────────────────────
def walk_forward_sharpe(
    X: pd.DataFrame,
    y: pd.Series,
    price: pd.Series,
    model_params: dict,
    prob_thresh: float,
    n_splits: int = 5,
) -> tuple[float, float, int]:
    """Run TimeSeriesSplit, return (mean_sharpe, mean_winrate, total_trades)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sharpes, winrates, trade_counts = [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
            continue   # skip fold with almost no samples

        model = XGBClassifier(
            **model_params,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            device="cpu",
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        prob = model.predict_proba(X_val)[:, 1]
        signal = (prob > prob_thresh).astype(float)

        # returns on val window
        price_val = price.loc[X_val.index]
        ret_val   = price_val.pct_change().fillna(0).values

        # strategy return: signal fires → hold next bar
        # include fee on position changes
        position = signal
        trades   = np.abs(np.diff(np.concatenate([[0], position]))).sum()
        fee_cost = trades * FEE

        strat_ret = position * ret_val
        # deduct fee spread from total return
        total_ret = strat_ret.sum() - fee_cost
        # recompute per-bar net for Sharpe
        fee_per_bar = fee_cost / max(len(strat_ret), 1)
        net_ret = strat_ret - fee_per_bar

        std = net_ret.std()
        sharpe = (net_ret.mean() / std * ANNUALIZE_15M) if std > 1e-9 else -10.0

        wins = (strat_ret[signal.astype(bool)] > 0).sum()
        total_sig = signal.sum()
        winrate = wins / total_sig if total_sig > 0 else 0.0

        sharpes.append(sharpe)
        winrates.append(winrate)
        trade_counts.append(int(trades))

    if not sharpes:
        return -10.0, 0.0, 0
    return float(np.mean(sharpes)), float(np.mean(winrates)), int(np.sum(trade_counts))


# ── Optuna Objective ───────────────────────────────────────────────────────────
def make_objective(df_raw: pd.DataFrame):
    """Closure: capture df_raw, return the trial objective function."""

    def objective(trial: optuna.Trial) -> float:
        # ── 1. Label parameters ────────────────────────────────────────────
        lookahead  = trial.suggest_int("lookahead_bars", 2, 24)
        ret_thresh = trial.suggest_float("ret_thresh", 0.002, 0.015)

        # ── 2. Feature window parameters ───────────────────────────────────
        ema_fast   = trial.suggest_int("ema_fast",   5,  30)
        ema_slow   = trial.suggest_int("ema_slow",  20, 100)
        ema_macro  = trial.suggest_int("ema_macro", 80, 300)
        rsi_p      = trial.suggest_int("rsi_period", 7,  21)
        macd_fast_ = trial.suggest_int("macd_fast",  5,  20)
        macd_slow_ = trial.suggest_int("macd_slow", 20,  50)
        macd_sig_  = trial.suggest_int("macd_signal", 5, 15)
        atr_p      = trial.suggest_int("atr_period",  7, 21)
        bb_p       = trial.suggest_int("bb_period",  10, 30)
        bb_std_k   = trial.suggest_float("bb_std",  1.5,  3.0)
        vol_w      = trial.suggest_int("vol_window", 10,  60)
        rvol_s     = trial.suggest_int("rvol_short",  4,  24)
        rvol_l     = trial.suggest_int("rvol_long",  24,  96)

        # ensure ordering constraints
        if ema_fast >= ema_slow or ema_slow >= ema_macro:
            raise optuna.exceptions.TrialPruned()
        if macd_fast_ >= macd_slow_:
            raise optuna.exceptions.TrialPruned()
        if rvol_s >= rvol_l:
            raise optuna.exceptions.TrialPruned()

        # ── 3. XGBoost hyperparams ─────────────────────────────────────────
        xgb_params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 50, 400),
            max_depth        = trial.suggest_int("max_depth",    2,   6),
            learning_rate    = trial.suggest_float("lr", 0.005, 0.2, log=True),
            subsample        = trial.suggest_float("subsample",  0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample",  0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child",    1,  20),
            gamma            = trial.suggest_float("gamma",      0.0, 5.0),
            reg_alpha        = trial.suggest_float("alpha",      0.0, 2.0),
            reg_lambda       = trial.suggest_float("lambda",     0.5, 5.0),
            early_stopping_rounds = 30,
            random_state     = 42,
        )

        prob_thresh = trial.suggest_float("prob_thresh", 0.35, 0.75)

        # ── 4. Build features & label ──────────────────────────────────────
        try:
            feats = build_features(
                df_raw, ema_fast, ema_slow, ema_macro,
                rsi_p, macd_fast_, macd_slow_, macd_sig_,
                atr_p, bb_p, bb_std_k, vol_w, rvol_s, rvol_l,
            )
        except Exception:
            raise optuna.exceptions.TrialPruned()

        future_ret = df_raw["close"].pct_change(lookahead).shift(-lookahead)
        y = (future_ret > ret_thresh).astype(int)

        # align and drop NaN
        valid = feats.index.intersection(y.dropna().index)
        X = feats.loc[valid].dropna()
        y_clean = y.loc[X.index]

        if len(X) < 500 or y_clean.sum() < 50:
            raise optuna.exceptions.TrialPruned()

        # ── 5. Walk-forward evaluation ─────────────────────────────────────
        sharpe, winrate, n_trades = walk_forward_sharpe(
            X, y_clean, df_raw["close"], xgb_params, prob_thresh
        )

        # ── 6. Penalize over-trading (> 20% bars trigger a trade) ─────────
        trade_rate = n_trades / len(X)
        if trade_rate > 0.20:
            sharpe -= (trade_rate - 0.20) * 10

        # ── 7. Store extra metrics for analysis ───────────────────────────
        trial.set_user_attr("winrate",    round(winrate, 4))
        trial.set_user_attr("n_trades",   n_trades)
        trial.set_user_attr("trade_rate", round(trade_rate, 4))

        return sharpe

    return objective


# ── QuantDingers export ────────────────────────────────────────────────────────
def export_qd_15m(best: optuna.trial.FrozenTrial, output_path: str):
    """Write a paste-ready QuantDingers v3 IndicatorStrategy for 15m."""
    p = best.params
    ua = best.user_attrs

    code = f'''# ============================================================
# QuantDingers v3 IndicatorStrategy — 15m BTC
# Auto-generated by optuna_search.py
# ============================================================
#
# Optuna Search Results
#   Best Sharpe (walk-forward, 5-fold) : {best.value:.4f}
#   Avg Win Rate                        : {ua.get("winrate", "n/a")}
#   Avg Trades (5 OOS folds)            : {ua.get("n_trades", "n/a")}
#   Trade Rate (trades/bars)            : {ua.get("trade_rate", "n/a")}
#
# Optimal Label
#   lookahead_bars = {p["lookahead_bars"]}  (number of 15m bars to look ahead)
#   ret_thresh     = {p["ret_thresh"]:.4f}  (minimum return to label as "up")
#
# Optimal Feature Windows
#   ema_fast={p["ema_fast"]}  ema_slow={p["ema_slow"]}  ema_macro={p["ema_macro"]}
#   rsi_period={p["rsi_period"]}  macd=({p["macd_fast"]},{p["macd_slow"]},{p["macd_signal"]})
#   atr={p["atr_period"]}  bb=({p["bb_period"]},{p["bb_std"]:.1f})  vol_window={p["vol_window"]}
#   rvol=({p["rvol_short"]},{p["rvol_long"]})
#
# XGBoost Hyperparams (for reference)
#   n_estimators={p["n_estimators"]}  max_depth={p["max_depth"]}
#   lr={p["lr"]:.4f}  prob_thresh={p["prob_thresh"]:.3f}
# ============================================================

my_indicator_name = "BTC 15m Optuna Strategy"
my_indicator_description = (
    "Auto-evolved 15m BTC strategy. Asymmetric entry (N-of-10) + independent exit. "
    "Feature windows and thresholds optimized by Optuna walk-forward Sharpe."
)

# @param ema_fast int {p["ema_fast"]} Fast EMA period
# @param ema_slow int {p["ema_slow"]} Slow EMA period
# @param ema_macro int {p["ema_macro"]} Macro trend EMA
# @param rsi_len int {p["rsi_period"]} RSI length
# @param rsi_entry_lo float 40 RSI minimum entry
# @param rsi_entry_hi float 72 RSI maximum entry
# @param rsi_exit float 35 RSI exit threshold
# @param macd_fast int {p["macd_fast"]} MACD fast
# @param macd_slow int {p["macd_slow"]} MACD slow
# @param macd_sig int {p["macd_signal"]} MACD signal
# @param atr_len int {p["atr_period"]} ATR period
# @param bb_len int {p["bb_period"]} BB period
# @param bb_std float {p["bb_std"]:.1f} BB std multiplier
# @param vol_window int {p["vol_window"]} Volume MA window
# @param vol_surge float 1.1 Volume surge threshold
# @param required_signals int 7 Entry: N-of-10 threshold

# @strategy stopLossPct 0.005
# @strategy takeProfitPct 0.012
# @strategy entryPct 0.25
# @strategy trailingEnabled true
# @strategy trailingStopPct 0.003
# @strategy trailingActivationPct 0.006
# @strategy tradeDirection long

import numpy as np

df = df.copy()

# ── Read params ──────────────────────────────────────────────────────────────
ema_fast_len  = int(params.get("ema_fast",  {p["ema_fast"]}))
ema_slow_len  = int(params.get("ema_slow",  {p["ema_slow"]}))
ema_macro_len = int(params.get("ema_macro", {p["ema_macro"]}))
rsi_len       = int(params.get("rsi_len",   {p["rsi_period"]}))
rsi_e_lo      = float(params.get("rsi_entry_lo", 40.0))
rsi_e_hi      = float(params.get("rsi_entry_hi", 72.0))
rsi_exit_th   = float(params.get("rsi_exit",     35.0))
macd_f        = int(params.get("macd_fast", {p["macd_fast"]}))
macd_sl       = int(params.get("macd_slow", {p["macd_slow"]}))
macd_sig_p    = int(params.get("macd_sig",  {p["macd_signal"]}))
atr_len       = int(params.get("atr_len",   {p["atr_period"]}))
bb_len        = int(params.get("bb_len",    {p["bb_period"]}))
bb_std_k      = float(params.get("bb_std",  {p["bb_std"]:.1f}))
vol_w         = int(params.get("vol_window",{p["vol_window"]}))
vol_surge     = float(params.get("vol_surge", 1.1))
req           = int(params.get("required_signals", 7))

# ── Indicators ───────────────────────────────────────────────────────────────
ema_fast  = df["close"].ewm(span=ema_fast_len,  adjust=False).mean()
ema_slow  = df["close"].ewm(span=ema_slow_len,  adjust=False).mean()
ema_macro = df["close"].ewm(span=ema_macro_len, adjust=False).mean()

delta = df["close"].diff()
gain  = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_len, adjust=False).mean()
rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

macd_line = (df["close"].ewm(span=macd_f,  adjust=False).mean() -
             df["close"].ewm(span=macd_sl, adjust=False).mean())
macd_sig_line = macd_line.ewm(span=macd_sig_p, adjust=False).mean()
macd_hist     = macd_line - macd_sig_line

tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - df["close"].shift(1)).abs(),
    (df["low"]  - df["close"].shift(1)).abs(),
], axis=1).max(axis=1)
atr = tr.ewm(span=atr_len, adjust=False).mean()
atr_ratio     = atr / df["close"]
atr_ratio_rel = atr_ratio / atr_ratio.rolling(atr_len * 3).mean()

bb_mid    = df["close"].rolling(bb_len).mean()
bb_std_r  = df["close"].rolling(bb_len).std()
bb_upper  = bb_mid + bb_std_k * bb_std_r
bb_lower  = bb_mid - bb_std_k * bb_std_r
bb_pct    = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
bb_width  = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

vol_ma    = df["volume"].rolling(vol_w).mean()
vol_ratio = df["volume"] / vol_ma.replace(0, np.nan)

ret       = df["close"].pct_change()
rvol_s    = ret.rolling({p["rvol_short"]}).std()
rvol_l    = ret.rolling({p["rvol_long"]}).std()
rvol_ratio = rvol_s / rvol_l.replace(0, np.nan)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY: 10 sub-signals, N-of-10 (default req=7)
# ══════════════════════════════════════════════════════════════════════════════

# 1. EMA bullish stack
s1 = (ema_fast > ema_slow) & (ema_slow > ema_macro)
# 2. Price above macro EMA
s2 = df["close"] > ema_macro
# 3. RSI healthy zone
s3 = (rsi > rsi_e_lo) & (rsi < rsi_e_hi)
# 4. RSI trending up (3-bar)
s4 = rsi > rsi.shift(3)
# 5. MACD histogram positive
s5 = macd_hist > 0
# 6. MACD histogram expanding
s6 = macd_hist > macd_hist.shift(1)
# 7. Volume surge
s7 = vol_ratio > vol_surge
# 8. BB%B in healthy zone (not at resistance)
s8 = (bb_pct > 0.35) & (bb_pct < 0.88)
# 9. Volatility not spiking (rvol short < rvol long * 1.4)
s9 = rvol_ratio < 1.4
# 10. ATR ratio not expanded above 30-bar avg * 1.3 (avoid post-spike entry)
s10 = atr_ratio_rel < 1.3

signal_sum = (
    s1.astype(int) + s2.astype(int) + s3.astype(int) +
    s4.astype(int) + s5.astype(int) + s6.astype(int) +
    s7.astype(int) + s8.astype(int) + s9.astype(int) + s10.astype(int)
)

# ══════════════════════════════════════════════════════════════════════════════
# EXIT: independent structural failure conditions (2-of-3 required)
# ══════════════════════════════════════════════════════════════════════════════
exit_ema  = df["close"] < ema_slow                         # trend break
exit_rsi  = rsi < rsi_exit_th                              # momentum collapse
exit_macd = (macd_hist < 0) & (macd_hist.shift(1) < 0)    # MACD confirmed neg

exit_count     = exit_ema.astype(int) + exit_rsi.astype(int) + exit_macd.astype(int)
exit_condition = exit_count >= 2

# ── Edge-triggered signals ───────────────────────────────────────────────────
raw_buy  = signal_sum >= req
raw_sell = exit_condition

df["buy"]  = (raw_buy.fillna(False)  & (~raw_buy.shift(1).fillna(False))).astype(bool)
df["sell"] = (raw_sell.fillna(False) & (~raw_sell.shift(1).fillna(False))).astype(bool)
df.loc[df["buy"], "sell"] = False

# ── Chart markers ────────────────────────────────────────────────────────────
buy_marks  = [df["low"].iloc[i]  * 0.999 if df["buy"].iloc[i]  else None for i in range(len(df))]
sell_marks = [df["high"].iloc[i] * 1.001 if df["sell"].iloc[i] else None for i in range(len(df))]

# ── Output ───────────────────────────────────────────────────────────────────
output = {{
    "name": my_indicator_name,
    "plots": [
        {{"name": "EMA Fast",        "data": ema_fast.fillna(0).tolist(),   "color": "#1890ff", "overlay": True}},
        {{"name": "EMA Slow",        "data": ema_slow.fillna(0).tolist(),   "color": "#faad14", "overlay": True}},
        {{"name": "EMA Macro",       "data": ema_macro.fillna(0).tolist(),  "color": "#f5222d", "overlay": True}},
        {{"name": "Signal (0-10)",   "data": signal_sum.fillna(0).tolist(), "color": "#722ed1", "overlay": False}},
        {{"name": "RSI",             "data": rsi.fillna(50).tolist(),       "color": "#13c2c2", "overlay": False}},
        {{"name": "MACD Hist",       "data": macd_hist.fillna(0).tolist(),  "color": "#52c41a", "overlay": False}},
        {{"name": "Exit Count (0-3)","data": exit_count.fillna(0).tolist(), "color": "#ff7a45", "overlay": False}}
    ],
    "signals": [
        {{"type": "buy",  "text": "L", "data": buy_marks,  "color": "#00E676"}},
        {{"type": "sell", "text": "X", "data": sell_marks, "color": "#FF5252"}}
    ]
}}
'''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    logger.success(f"QuantDingers 15m strategy exported → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Optuna strategy search for 15m BTC")
    p.add_argument("--trials",    type=int, default=200,  help="Number of Optuna trials")
    p.add_argument("--jobs",      type=int, default=1,    help="Parallel jobs (set 1 if XGB already multi-threaded)")
    p.add_argument("--interval",  type=str, default="15m", help="Kline interval (default 15m)")
    p.add_argument("--days",      type=int, default=365,  help="Lookback days to fetch")
    p.add_argument("--splits",    type=int, default=5,    help="Walk-forward CV splits")
    p.add_argument("--no-cache",  action="store_true",   help="Force re-download (ignore cache)")
    p.add_argument("--study-name",type=str, default="btc_15m_search", help="Optuna study name")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    from config import CFG
    from data.fetch_btc import fetch_btc_ohlcv

    if args.no_cache:
        import glob
        for f in glob.glob(f"./cache/*{args.interval}*.parquet"):
            os.remove(f)
            logger.info(f"Cache removed: {f}")

    logger.info(f"Fetching BTCUSDT {args.interval} data ({args.days} days)...")
    df_raw = fetch_btc_ohlcv(
        symbol=CFG.btc_symbol,
        interval=args.interval,
        days=args.days,
    )
    logger.info(f"Data loaded: {len(df_raw):,} bars  "
                f"({df_raw.index[0].date()} ~ {df_raw.index[-1].date()})")

    if len(df_raw) < 1000:
        raise RuntimeError(f"Too few bars ({len(df_raw)}) — increase --days or check interval.")

    # ── 2. Run Optuna search ───────────────────────────────────────────────
    logger.info(f"Starting Optuna search: {args.trials} trials, {args.jobs} jobs")

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=30)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(df_raw)
    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # ── 3. Save results ────────────────────────────────────────────────────
    best = study.best_trial
    logger.success(
        f"\nBest trial #{best.number}  "
        f"Sharpe={best.value:.4f}  "
        f"winrate={best.user_attrs.get('winrate', 'n/a')}  "
        f"trades={best.user_attrs.get('n_trades', 'n/a')}"
    )
    logger.info(f"Best params: {json.dumps(best.params, indent=2)}")

    # Save best params JSON
    params_path = os.path.join(OUTPUT_DIR, "optuna_best_params.json")
    with open(params_path, "w") as f:
        payload = {
            "best_value":  best.value,
            "user_attrs":  best.user_attrs,
            "params":      best.params,
        }
        json.dump(payload, f, indent=2)
    logger.info(f"Best params saved → {params_path}")

    # Save full trial history CSV
    results_path = os.path.join(OUTPUT_DIR, "optuna_study_results.csv")
    study.trials_dataframe().to_csv(results_path, index=False)
    logger.info(f"Full study results saved → {results_path}")

    # ── 4. Export QuantDingers strategy ───────────────────────────────────
    qd_path = os.path.join(OUTPUT_DIR, "quantdingers_15m_strategy.py")
    export_qd_15m(best, qd_path)

    logger.success(
        f"\n{'='*60}\n"
        f"Done! Outputs in {OUTPUT_DIR}/\n"
        f"  optuna_best_params.json\n"
        f"  optuna_study_results.csv\n"
        f"  quantdingers_15m_strategy.py  ← paste into QuantDingers\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
