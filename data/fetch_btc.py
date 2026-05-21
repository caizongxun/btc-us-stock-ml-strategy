"""Fetch BTC OHLCV from Binance public API (no key needed for historical klines)"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from config import CFG
import os

BASE_URL = "https://api.binance.com/api/v3/klines"

def fetch_btc_ohlcv(symbol: str = CFG.btc_symbol, interval: str = CFG.btc_interval, days: int = CFG.lookback_days) -> pd.DataFrame:
    """Download BTC klines. Returns DataFrame with columns: open high low close volume."""
    cache_path = os.path.join(CFG.data_dir, f"{symbol}_{interval}_{days}.parquet")
    os.makedirs(CFG.data_dir, exist_ok=True)

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded BTC from cache: {len(df)} rows")
        return df

    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    limit = 1000
    all_klines = []

    while start_time < end_time:
        params = dict(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time, limit=limit)
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        klines = r.json()
        if not klines:
            break
        all_klines.extend(klines)
        start_time = klines[-1][0] + 1
        logger.info(f"Fetched {len(all_klines)} BTC candles so far...")

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(all_klines, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    num_cols = ["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df = df.set_index("date")[["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]]
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_parquet(cache_path)
    logger.info(f"BTC fetched: {len(df)} rows")
    return df
