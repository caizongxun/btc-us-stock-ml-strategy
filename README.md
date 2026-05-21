# BTC + US Stock ML Strategy

A full machine learning pipeline for generating **binary trading rules** and **ML-driven signals** targeting BTC and US equities (SPY/QQQ).

Designed for **QuantDingers** integration — outputs `+1 / 0 / -1` signals directly.

---

## Architecture

```
btc-us-stock-ml-strategy/
├── data/
│   ├── fetch_btc.py          # Binance OHLCV
│   ├── fetch_stocks.py       # yfinance SPY/QQQ/VIX
│   └── fetch_onchain.py      # Fear & Greed + SOPR proxy
├── features/
│   ├── price_features.py     # momentum, returns, drawdown
│   ├── volatility_features.py# ATR, rolling std, VIX regime
│   ├── cross_asset_features.py# BTC-SPY correlation, beta, divergence
│   ├── technical_features.py # RSI, MACD, BB, MFI, etc.
│   ├── regime_features.py    # HMM/GMM market regime
│   └── build_features.py     # master feature assembler
├── models/
│   ├── xgboost_model.py      # XGBoost classifier
│   ├── lightgbm_model.py     # LightGBM classifier
│   ├── regime_detector.py    # HMM regime segmentation
│   └── binary_rule_miner.py  # Decision tree rule extractor
├── strategy/
│   ├── signal_generator.py   # model -> +1/0/-1 signal
│   ├── multi_signal.py       # N-of-M signal voting
│   └── quantdingers_export.py# export strategy code
├── backtest/
│   ├── backtester.py         # vectorbt-based backtester
│   └── evaluate.py           # Sharpe, Calmar, drawdown
├── config.py                 # global settings
├── main.py                   # full pipeline runner
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --asset BTC --target_days 3 --mode full
```

## Modes
- `full` — train XGBoost + LightGBM + Regime + extract binary rules
- `rules_only` — decision tree rule mining only
- `regime_only` — HMM regime detection only
- `export` — export QuantDingers-compatible strategy
