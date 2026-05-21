"""Evaluation metrics and plotting"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import CFG

def evaluate_signal(price: pd.Series, signal: pd.Series, label: str = "strategy") -> dict:
    ret = price.pct_change().fillna(0)
    pos = signal.shift(1).fillna(0)
    strategy_ret = pos * ret
    cum = (1 + strategy_ret).cumprod()
    bh = (1 + ret).cumprod()

    sharpe = strategy_ret.mean() / (strategy_ret.std() + 1e-9) * np.sqrt(252)
    sortino_denom = strategy_ret[strategy_ret < 0].std() + 1e-9
    sortino = strategy_ret.mean() / sortino_denom * np.sqrt(252)
    max_dd = (cum / cum.cummax() - 1).min()
    total_ret = cum.iloc[-1] - 1
    calmar = total_ret / (abs(max_dd) + 1e-9)
    win_rate = (strategy_ret[pos != 0] > 0).mean()
    n_trades = (signal.diff() != 0).sum()

    metrics = {
        "label": label,
        "total_return": round(total_ret, 4),
        "bh_return": round(bh.iloc[-1] - 1, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": int(n_trades),
    }

    os.makedirs(CFG.output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    cum.plot(ax=axes[0], label=label, color="steelblue")
    bh.plot(ax=axes[0], label="Buy & Hold", color="gray", linestyle="--")
    axes[0].set_title(f"Equity Curve — {label}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    drawdown = cum / cum.cummax() - 1
    drawdown.plot(ax=axes[1], color="red", alpha=0.6)
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
    axes[1].set_title("Drawdown")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(CFG.output_dir, f"{label}_backtest.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return metrics
