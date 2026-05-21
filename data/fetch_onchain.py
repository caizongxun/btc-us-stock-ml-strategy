"""Fetch on-chain/sentiment proxies: Fear & Greed Index, BTC dominance"""
import requests
import pandas as pd
from loguru import logger
from config import CFG
import os

def fetch_fear_greed(days: int = CFG.lookback_days) -> pd.DataFrame:
    """Alternative.me Fear & Greed Index — free, no key required"""
    cache_path = os.path.join(CFG.data_dir, f"fear_greed_{days}.parquet")
    os.makedirs(CFG.data_dir, exist_ok=True)
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("date")[["value","value_classification"]]
    df["fg_value"] = df["value"].astype(float)
    df["fg_extreme_fear"] = (df["fg_value"] < 25).astype(int)
    df["fg_extreme_greed"] = (df["fg_value"] > 75).astype(int)
    df = df.drop(columns=["value","value_classification"]).sort_index()
    df.to_parquet(cache_path)
    logger.info(f"Fear & Greed fetched: {len(df)} rows")
    return df
