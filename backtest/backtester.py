"""Vectorbt-based backtester for signal evaluation"""
import numpy as np
import pandas as pd
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
from loguru import logger

def run_backtest(price: pd.Series, signal: pd.Series, init_cash: float = 10_000) -> dict:
    """
    price: close price Series aligned with signal
    signal: +1 (long), 0 (flat), -1 (short)
    Returns dict of performance metrics
    """
    if not HAS_VBT:
        logger.warning("vectorbt not installed, running simple backtest")
        return _simple_backtest(price, signal, init_cash)

    entries = signal == 1
    exits = signal != 1
    short_entries = signal == -1
    short_exits = signal != -1

    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        fees=0.001,
        slippage=0.001,
    )
    stats = pf.stats()
    logger.info(f"Backtest stats:\n{stats}")
    return {"portfolio": pf, "stats": stats}

def _simple_backtest(price: pd.Series, signal: pd.Series, init_cash: float) -> dict:
    """Fallback simple backtest if vectorbt not available."""
    ret = price.pct_change().fillna(0)
    pos = signal.shift(1).fillna(0)  # lag to avoid lookahead
    strategy_ret = pos * ret
    cum_ret = (1 + strategy_ret).cumprod()
    bh_ret = (1 + ret).cumprod()
    sharpe = strategy_ret.mean() / (strategy_ret.std() + 1e-9) * np.sqrt(252)
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    total_ret = cum_ret.iloc[-1] - 1
    calmar = total_ret / (abs(max_dd) + 1e-9)
    result = {
        "total_return": round(total_ret, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "bh_total_return": round(bh_ret.iloc[-1] - 1, 4),
        "cum_returns": cum_ret,
    }
    logger.info(f"Simple backtest: {result}")
    return result
