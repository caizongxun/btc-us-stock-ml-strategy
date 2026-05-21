"""Technical indicator features: RSI, MACD, Bollinger Bands, MFI, Stochastic, etc."""
import pandas as pd
import numpy as np

def _rsi(series, n):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _bollinger(series, n=20, k=2):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    bandwidth = (upper - lower) / (ma + 1e-9)
    return pct_b, bandwidth

def _stochastic(high, low, close, k=14, d=3):
    low_min = low.rolling(k).min()
    high_max = high.rolling(k).max()
    k_pct = 100 * (close - low_min) / (high_max - low_min + 1e-9)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct

def _mfi(high, low, close, volume, n=14):
    typical = (high + low + close) / 3
    mf = typical * volume
    pos_mf = mf.where(typical > typical.shift(1), 0).rolling(n).sum()
    neg_mf = mf.where(typical < typical.shift(1), 0).rolling(n).sum()
    return 100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))

def _cci(high, low, close, n=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md + 1e-9)

def add_technical_features(df: pd.DataFrame, prefix: str = "btc") -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # RSI
    for n in [7, 14, 21, 30]:
        df[f"{prefix}_rsi_{n}"] = _rsi(c, n) / 100  # normalize 0-1

    # RSI divergence (overbought/oversold flags)
    df[f"{prefix}_rsi14_ob"] = (_rsi(c, 14) > 70).astype(int)
    df[f"{prefix}_rsi14_os"] = (_rsi(c, 14) < 30).astype(int)

    # MACD (normalized by price)
    for fast, slow, sig in [(12, 26, 9), (5, 13, 6)]:
        macd, signal, hist = _macd(c, fast, slow, sig)
        tag = f"{fast}_{slow}"
        df[f"{prefix}_macd_hist_{tag}"] = hist / (c + 1e-9)
        df[f"{prefix}_macd_cross_{tag}"] = (macd > signal).astype(int)
        df[f"{prefix}_macd_hist_dir_{tag}"] = (hist > hist.shift(1)).astype(int)

    # Bollinger Bands
    for n in [14, 20, 30]:
        pct_b, bw = _bollinger(c, n)
        df[f"{prefix}_bb_pctb_{n}"] = pct_b
        df[f"{prefix}_bb_bw_{n}"] = bw
        df[f"{prefix}_bb_squeeze_{n}"] = (bw < bw.rolling(90).quantile(0.20)).astype(int)

    # Stochastic
    k, d = _stochastic(h, l, c)
    df[f"{prefix}_stoch_k"] = k / 100
    df[f"{prefix}_stoch_d"] = d / 100
    df[f"{prefix}_stoch_cross"] = (k > d).astype(int)

    # MFI
    df[f"{prefix}_mfi_14"] = _mfi(h, l, c, v, 14) / 100

    # CCI
    df[f"{prefix}_cci_20"] = _cci(h, l, c, 20) / 200  # normalize
    df[f"{prefix}_cci_40"] = _cci(h, l, c, 40) / 200

    # EMA slopes (trend direction)
    for n in [9, 21, 50, 100, 200]:
        ema = c.ewm(span=n, adjust=False).mean()
        df[f"{prefix}_ema{n}_slope"] = (ema / ema.shift(5) - 1)  # 5-day EMA slope
        df[f"{prefix}_price_vs_ema{n}"] = (c / ema) - 1           # price premium over EMA

    # EMA crossover signals
    ema9 = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    ema50 = c.ewm(span=50).mean()
    ema200 = c.ewm(span=200).mean()
    df[f"{prefix}_ema9_above_21"] = (ema9 > ema21).astype(int)
    df[f"{prefix}_ema21_above_50"] = (ema21 > ema50).astype(int)
    df[f"{prefix}_ema50_above_200"] = (ema50 > ema200).astype(int)  # golden cross proxy
    df[f"{prefix}_trend_score"] = (
        df[f"{prefix}_ema9_above_21"] +
        df[f"{prefix}_ema21_above_50"] +
        df[f"{prefix}_ema50_above_200"]
    )  # 0-3 trend alignment score

    # OBV slope
    obv = (np.sign(c.diff()) * v).cumsum()
    df[f"{prefix}_obv_slope_14"] = (obv / obv.shift(14) - 1)

    return df
