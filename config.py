from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- Assets ---
    btc_symbol: str = "BTCUSDT"
    stock_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "^VIX", "GLD", "TLT"])
    btc_interval: str = "4h"          # 4h candles
    lookback_days: int = 1500

    # --- Labels ---
    target_asset: str = "BTC"
    target_days: int = 14             # 4h bars: 14 bars ≈ 2.3 days forward
    long_threshold: float = 0.020
    short_threshold: float = 0.020

    # --- Feature windows (in bars, 4h units) ---
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
    long_prob_threshold: float = 0.52
    short_prob_threshold: float = 0.48

    # --- Multi-signal voting ---
    # n_of_m_threshold=4 with 9 sub-signals:
    #   long  = vote >= 4  (~44% of signals need to agree)
    #   flat  = vote <  4  (stay out)
    # No short signal — see multi_signal.py for rationale.
    n_of_m_threshold: int = 4

    # --- Regime ---
    n_regimes: int = 3

    # --- Paths ---
    data_dir: str = "./cache"
    model_dir: str = "./saved_models"
    output_dir: str = "./outputs"

CFG = Config()
