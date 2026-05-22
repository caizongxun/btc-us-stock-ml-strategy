from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- Assets ---
    btc_symbol: str = "BTCUSDT"
    stock_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "^VIX", "GLD", "TLT"])
    btc_interval: str = "4h"
    lookback_days: int = 1500

    # --- Labels ---
    target_asset: str = "BTC"
    target_days: int = 14
    long_threshold: float = 0.020
    short_threshold: float = 0.020

    # --- Feature windows ---
    short_windows: List[int] = field(default_factory=lambda: [6, 12, 24])
    mid_windows: List[int]   = field(default_factory=lambda: [42, 60, 84])
    long_windows: List[int]  = field(default_factory=lambda: [180, 252, 360])

    # --- Feature selection ---
    top_k_features: int = 50

    # --- Model ---
    n_trials: int = 50
    cv_splits: int = 5
    early_stopping_rounds: int = 50
    test_size: float = 0.2

    # --- Signal probability thresholds ---
    # Set to None → models will auto-sweep calib slice for best F1-class1 threshold.
    # Set to a float (e.g. 0.35) to override auto-sweep.
    long_prob_threshold: float = None   # None = auto
    short_prob_threshold: float = None  # None = auto

    # Threshold sweep range when long_prob_threshold is None
    threshold_sweep_min: float = 0.25
    threshold_sweep_max: float = 0.60
    threshold_sweep_steps: int = 36

    # --- Multi-signal voting ---
    # Lowered from 5 → 4: ML signal fire rate is only 5%, so requiring 6-of-10
    # made it near-irrelevant. 4-of-10 keeps the composite signal more balanced.
    n_of_m_threshold: int = 4

    # --- Regime ---
    n_regimes: int = 3

    # --- Paths ---
    data_dir: str = "./cache"
    model_dir: str = "./saved_models"
    output_dir: str = "./outputs"

CFG = Config()
