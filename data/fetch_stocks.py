"""Fetch US stock / ETF / VIX data via yfinance"""
import yfinance as yf
import pandas as pd
from loguru import logger
from config import CFG
import os

def fetch_stocks(symbols: list = None, days: int = CFG.lookback_days) -> dict:
    """Returns dict of symbol -> OHLCV DataFrame"""
    symbols = symbols or CFG.stock_symbols
    results = {}
    os.makedirs(CFG.data_dir, exist_ok=True)

    for sym in symbols:
        cache_path = os.path.join(CFG.data_dir, f"{sym.replace('^','')}_daily_{days}.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            logger.info(f"{sym} loaded from cache: {len(df)} rows")
        else:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{days}d", interval="1d", auto_adjust=True)
            df.index = df.index.tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            df = df[["open","high","low","close","volume"]]
            df.to_parquet(cache_path)
            logger.info(f"{sym} fetched: {len(df)} rows")
        results[sym] = df
    return results
