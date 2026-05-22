"""LightGBM classifier with Optuna (average_precision objective) + calibration.

Changes vs previous version:
  1. Optuna objective: roc_auc → average_precision (PR-AUC).
  2. Auto threshold sweep on calibration slice (same logic as XGB).
  3. DataFrame passed throughout to avoid CalibratedClassifierCV feature-name warnings.
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score,
)
from loguru import logger
import joblib, os
from config import CFG

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep thresholds on the given slice; return argmax F1 for class-1."""
    thresholds = np.linspace(
        CFG.threshold_sweep_min,
        CFG.threshold_sweep_max,
        CFG.threshold_sweep_steps,
    )
    best_t, best_f1 = thresholds[0], -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logger.info(f"LGB threshold sweep → best_t={best_t:.3f}  F1-class1={best_f1:.4f}")
    return float(best_t)


def train_lightgbm(df: pd.DataFrame, label_col: str = "y_binary"):
    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]

    # Keep as DataFrame — preserves feature names across CalibratedClassifierCV folds
    X_df = df[feature_cols].copy()
    y    = df[label_col].values

    split_idx = int(len(X_df) * (1 - CFG.test_size))
    X_train_df, X_test_df = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_train,    y_test     = y[:split_idx],          y[split_idx:]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    logger.info(f"LGB scale_pos_weight (auto): {spw:.3f}  (neg={neg}, pos={pos})")
    logger.info(
        f"LGB label dist  train: pos={pos} ({100*pos/(neg+pos):.1f}%)  "
        f"test: pos={(y_test==1).sum()} ({100*(y_test==1).mean():.1f}%)"
    )

    tscv = TimeSeriesSplit(n_splits=CFG.cv_splits)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  spw,
            "is_unbalance":      False,
            "random_state":      42,
            "n_jobs":            1,
            "verbose":           -1,
        }
        m = lgb.LGBMClassifier(**params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            scores = cross_val_score(
                m, X_train_df, y_train,
                cv=tscv, scoring="average_precision", n_jobs=1,
            )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)
    logger.info(f"LightGBM best Avg-Precision (CV): {study.best_value:.4f}")

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "is_unbalance": False,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    calib_split = int(len(X_train_df) * 0.8)
    X_fit_df,   X_calib_df = X_train_df.iloc[:calib_split], X_train_df.iloc[calib_split:]
    y_fit,      y_calib    = y_train[:calib_split],          y_train[calib_split:]

    base_model = lgb.LGBMClassifier(**best_params)
    base_model.fit(
        X_fit_df, y_fit,
        eval_set=[(X_calib_df, y_calib)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    calib_tscv = TimeSeriesSplit(n_splits=3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=calib_tscv)
        model.fit(X_calib_df, y_calib)
    logger.info("CalibratedClassifierCV (isotonic, cv=3-fold-ts) fitted on calib slice")

    # --- Auto threshold sweep on calib slice ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        calib_prob = model.predict_proba(X_calib_df)[:, 1]
    if CFG.long_prob_threshold is None:
        threshold = _find_best_threshold(y_calib, calib_prob)
    else:
        threshold = CFG.long_prob_threshold
        logger.info(f"LGB using fixed threshold={threshold:.3f} (from CFG)")

    # --- Final test evaluation ---
    y_prob = model.predict_proba(X_test_df)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    logger.info(f"LGB calibrated  ROC-AUC={auc:.4f}  Avg-Precision={ap:.4f}")
    logger.info(
        f"LGB prob stats: min={y_prob.min():.3f} max={y_prob.max():.3f} "
        f"mean={y_prob.mean():.3f} >0.5: {(y_prob>0.5).sum()}"
    )

    y_pred = (y_prob >= threshold).astype(int)
    logger.info(f"LGB threshold={threshold:.3f}  predicted positives: {y_pred.sum()} / {len(y_pred)}")

    if y_pred.sum() == 0:
        softer = float(np.percentile(y_prob, 80))
        y_pred = (y_prob >= softer).astype(int)
        logger.warning(f"Recall(1)=0 at {threshold:.3f} — fallback to p80={softer:.3f}")

    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    importance = pd.Series(
        base_model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 LGB features:\n{importance.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "lgbm_model.pkl"))
    return model, importance
