from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- Assets ---
    btc_symbol: str = "BTCUSDT"
    stock_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "^VIX", "GLD", "TLT"])
    btc_interval: str = "1d"          # 1d candles
    lookback_days: int = 1500

    # --- Labels ---
    target_asset: str = "BTC"         # BTC or SPY or QQQ
    target_days: int = 3              # forward return window
    long_threshold: float = 0.015     # +1.5% = long
    short_threshold: float = -0.015   # -1.5% = short

    # --- Feature windows ---
    short_windows: List[int] = field(default_factory=lambda: [3, 5, 7])
    mid_windows: List[int] = field(default_factory=lambda: [14, 20, 30])
    long_windows: List[int] = field(default_factory=lambda: [60, 90, 120])

    # --- Model ---
    n_trials: int = 50                # Optuna trials
    cv_splits: int = 5                # TimeSeriesSplit folds
    early_stopping_rounds: int = 50
    test_size: float = 0.2

    # --- Signal ---
    long_prob_threshold: float = 0.60
    short_prob_threshold: float = 0.40
    n_of_m_threshold: int = 3         # multi-signal: need 3 of M to fire

    # --- Regime ---
    n_regimes: int = 3                # bull / sideways / bear

    # --- Paths ---
    data_dir: str = "./cache"
    model_dir: str = "./saved_models"
    output_dir: str = "./outputs"

CFG = Config()
