from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # --- Assets ---
    btc_symbol: str = "BTCUSDT"
    stock_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "^VIX", "GLD", "TLT"])
    btc_interval: str = "4h"          # 4h candles — ~6x more samples than 1d
    lookback_days: int = 1500

    # --- Labels ---
    target_asset: str = "BTC"         # BTC or SPY or QQQ
    target_days: int = 14             # 4h bars: 14 bars ≈ 2.3 days forward
    long_threshold: float = 0.020     # +2.0% = long  (wider band for 4h noise)
    short_threshold: float = -0.020   # -2.0% = short

    # --- Feature windows (in bars, 4h units) ---
    short_windows: List[int] = field(default_factory=lambda: [6, 12, 24])   # 1d, 2d, 4d
    mid_windows: List[int]   = field(default_factory=lambda: [42, 60, 84])  # 7d, 10d, 14d
    long_windows: List[int]  = field(default_factory=lambda: [180, 252, 360])  # 30d, 42d, 60d

    # --- Feature selection ---
    top_k_features: int = 50          # keep top-K by SHAP importance

    # --- Model ---
    n_trials: int = 50                # Optuna trials
    cv_splits: int = 5                # TimeSeriesSplit folds
    early_stopping_rounds: int = 50
    test_size: float = 0.2

    # --- Signal probability thresholds ---
    # Lower than default 0.60 to get non-zero recall on minority class.
    # The soft-fallback in model files will drop further to 0.45 if still 0.
    long_prob_threshold: float = 0.52
    short_prob_threshold: float = 0.48

    # --- Multi-signal voting ---
    # 3-of-8: fires when at least 3 sub-signals agree.
    # Dynamic logic in multi_signal.py will lower this if fewer signals are available.
    n_of_m_threshold: int = 3

    # --- Regime ---
    n_regimes: int = 3                # bull / sideways / bear

    # --- Paths ---
    data_dir: str = "./cache"
    model_dir: str = "./saved_models"
    output_dir: str = "./outputs"

CFG = Config()
