"""Fetch BTC OHLCV from data.binance.vision (public S3, no geo-block, no API key needed)

data.binance.vision URL pattern:
  https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY-MM-DD}.zip
  https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY-MM}.zip

Strategy:
  1. Download monthly ZIPs for all complete months in the lookback window
  2. Download daily ZIPs for the current (incomplete) month
  3. Concatenate, deduplicate, cache as parquet
"""
import io
import os
import zipfile
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import requests
from loguru import logger

from config import CFG

BASE = "https://data.binance.vision/data/spot"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
NUM_COLS = ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]


def _download_zip(url: str) -> pd.DataFrame | None:
    """Download a single ZIP from data.binance.vision and return a DataFrame."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLS)
        return df
    except Exception as e:
        logger.warning(f"Failed {url}: {e}")
        return None


def fetch_btc_ohlcv(
    symbol: str = CFG.btc_symbol,
    interval: str = CFG.btc_interval,
    days: int = CFG.lookback_days,
) -> pd.DataFrame:
    """Download BTC klines via data.binance.vision.
    Returns DataFrame indexed by date with columns: open high low close volume taker_buy_base taker_buy_quote
    """
    cache_path = os.path.join(CFG.data_dir, f"{symbol}_{interval}_{days}_vision.parquet")
    os.makedirs(CFG.data_dir, exist_ok=True)

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded BTC from cache: {len(df)} rows")
        return df

    today = date.today()
    start_date = today - timedelta(days=days)
    frames = []

    # --- Monthly ZIPs (faster, one file per month) ---
    cur = date(start_date.year, start_date.month, 1)
    # last complete month = last month
    last_complete = date(today.year, today.month, 1) - relativedelta(months=1)

    while cur <= last_complete:
        ym = cur.strftime("%Y-%m")
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        logger.info(f"Downloading monthly: {ym}")
        df_month = _download_zip(url)
        if df_month is not None:
            frames.append(df_month)
        cur += relativedelta(months=1)

    # --- Daily ZIPs for current (incomplete) month ---
    cur_day = date(today.year, today.month, 1)
    while cur_day < today:
        ymd = cur_day.strftime("%Y-%m-%d")
        url = f"{BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{ymd}.zip"
        df_day = _download_zip(url)
        if df_day is not None:
            frames.append(df_day)
        cur_day += timedelta(days=1)

    if not frames:
        raise RuntimeError("No data downloaded from data.binance.vision — check symbol/interval.")

    df = pd.concat(frames, ignore_index=True)
    df[NUM_COLS] = df[NUM_COLS].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("date")[["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Filter to requested lookback
    cutoff = pd.Timestamp(today - timedelta(days=days))
    df = df[df.index >= cutoff]

    df.to_parquet(cache_path)
    logger.info(f"BTC fetched via data.binance.vision: {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df
