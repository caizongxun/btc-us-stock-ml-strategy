"""Market regime features using Hidden Markov Model (HMM)"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from config import CFG
from loguru import logger

def detect_regime(df: pd.DataFrame, prefix: str = "btc", n_states: int = CFG.n_regimes) -> pd.DataFrame:
    """
    Use a Gaussian HMM to detect market regimes.
    Features fed to HMM: return, volatility, volume z-score.
    Returns original df with regime columns appended.
    """
    ret = df["close"].pct_change().fillna(0)
    vol = ret.rolling(14).std().fillna(0)
    v_z = (df["volume"] - df["volume"].rolling(30).mean()) / (df["volume"].rolling(30).std() + 1e-9)
    v_z = v_z.fillna(0)

    X = np.column_stack([ret.values, vol.values, v_z.values])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
    model.fit(X_scaled)
    states = model.predict(X_scaled)

    # Order states by mean return (0=bear, 1=sideways, 2=bull)
    state_means = [ret[states == s].mean() for s in range(n_states)]
    order = np.argsort(state_means)  # ascending
    remap = {old: new for new, old in enumerate(order)}
    states_ordered = np.vectorize(remap.get)(states)

    df[f"{prefix}_regime"] = states_ordered
    df[f"{prefix}_regime_bull"] = (states_ordered == n_states - 1).astype(int)
    df[f"{prefix}_regime_bear"] = (states_ordered == 0).astype(int)
    df[f"{prefix}_regime_sideways"] = (states_ordered == 1).astype(int)

    # Regime persistence: how many days in current regime
    regime_series = pd.Series(states_ordered, index=df.index)
    df[f"{prefix}_regime_days"] = regime_series.groupby((regime_series != regime_series.shift()).cumsum()).cumcount() + 1

    logger.info(f"HMM regime distribution: {pd.Series(states_ordered).value_counts().to_dict()}")
    return df
