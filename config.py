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
    # Lowered to 0.45: calibrated prob rarely exceeds 0.52 with isotonic calibration
    # on imbalanced data (pos=14%). 0.45 captures the top decile of predictions.
    long_prob_threshold: float = 0.45
    short_prob_threshold: float = 0.45

    # --- Multi-signal voting ---
    # Base: 5-of-9. If ml_signal fires, effective threshold drops to 4-of-9.
    n_of_m_threshold: int = 5

    # --- Regime ---
    n_regimes: int = 3

    # --- Paths ---
    data_dir: str = "./cache"
    model_dir: str = "./saved_models"
    output_dir: str = "./outputs"

CFG = Config()
