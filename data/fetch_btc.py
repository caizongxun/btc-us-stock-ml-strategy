"""Fetch BTC OHLCV from data.binance.vision (public S3, no geo-block, no API key needed)

data.binance.vision URL pattern:
  monthly: .../spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY-MM}.zip
  daily:   .../spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY-MM-DD}.zip
"""
import io
import os
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from loguru import logger

from config import CFG

BASE = "https://data.binance.vision/data/spot"

# Binance kline CSV columns (no header in file)
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
NUM_COLS = ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]
KEEP_COLS = ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]


def _download_zip(url: str) -> pd.DataFrame | None:
    """Download a single ZIP and return a raw DataFrame (no index set yet)."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=KLINE_COLS,
                    dtype=str,          # read everything as str first to avoid overflow
                )
        return df
    except Exception as e:
        logger.warning(f"Failed {url}: {e}")
        return None


def _parse_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate raw frames, parse types, build DatetimeIndex."""
    df = pd.concat(frames, ignore_index=True)

    # open_time is ms epoch; cast to int64 safely
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    df["open_time"] = df["open_time"].astype("int64")

    # Convert to datetime using ms unit with pandas >= 2.0 safe path
    # Use unit='ms' on integer series — avoids nanosecond overflow from close_time bleed
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)

    # Numeric columns
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("date")[KEEP_COLS]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()
    return df


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

    # --- Monthly ZIPs (one file per complete month) ---
    cur = date(start_date.year, start_date.month, 1)
    last_complete = date(today.year, today.month, 1) - relativedelta(months=1)

    while cur <= last_complete:
        ym = cur.strftime("%Y-%m")
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        logger.info(f"Downloading monthly: {ym}")
        df_part = _download_zip(url)
        if df_part is not None:
            frames.append(df_part)
        cur += relativedelta(months=1)

    # --- Daily ZIPs for current incomplete month ---
    cur_day = date(today.year, today.month, 1)
    while cur_day < today:
        ymd = cur_day.strftime("%Y-%m-%d")
        url = f"{BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{ymd}.zip"
        df_part = _download_zip(url)
        if df_part is not None:
            frames.append(df_part)
        cur_day += timedelta(days=1)

    if not frames:
        raise RuntimeError("No data downloaded from data.binance.vision — check symbol/interval.")

    df = _parse_frames(frames)

    # Filter to requested lookback window
    cutoff = pd.Timestamp(today - timedelta(days=days))
    df = df[df.index >= cutoff]

    df.to_parquet(cache_path)
    logger.info(
        f"BTC fetched via data.binance.vision: {len(df)} rows "
        f"({df.index[0].date()} ~ {df.index[-1].date()})"
    )
    return df
