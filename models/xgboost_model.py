"""XGBoost classifier with Optuna (average_precision objective) + SHAP.

Changes vs previous version:
  1. Optuna objective: roc_auc → average_precision (PR-AUC).
  2. Auto threshold sweep on the calibration slice: find the threshold
     that maximises F1 for class-1.
  3. Returns best_threshold so callers can pass it to signal_generator.
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
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

_FALLBACK_THRESHOLD = 0.45


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, label: str = "") -> float:
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
    logger.info(f"{label}threshold sweep → best_t={best_t:.3f}  F1-class1={best_f1:.4f}")
    return float(best_t)


def train_xgboost(df: pd.DataFrame, label_col: str = "y_binary"):
    """
    Returns: (model, shap_importance, (X_test, y_test, y_prob), best_threshold)
    """
    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    X = df[feature_cols].values
    y = df[label_col].values

    split_idx = int(len(X) * (1 - CFG.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    logger.info(f"XGB scale_pos_weight (auto): {spw:.3f}  (neg={neg}, pos={pos})")
    logger.info(
        f"XGB label dist  train: pos={pos} ({100*pos/(neg+pos):.1f}%)  "
        f"test: pos={(y_test==1).sum()} ({100*(y_test==1).mean():.1f}%)"
    )

    tscv = TimeSeriesSplit(n_splits=CFG.cv_splits)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": spw,
            "eval_metric":      "logloss",
            "random_state":     42,
            "n_jobs":           1,
        }
        m = xgb.XGBClassifier(**params)
        scores = cross_val_score(
            m, X_train, y_train,
            cv=tscv, scoring="average_precision", n_jobs=1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)
    logger.info(f"XGBoost best Avg-Precision (CV): {study.best_value:.4f}")

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    calib_split = int(len(X_train) * 0.8)
    X_fit,  X_calib = X_train[:calib_split], X_train[calib_split:]
    y_fit,  y_calib = y_train[:calib_split], y_train[calib_split:]

    base_model = xgb.XGBClassifier(**best_params)
    base_model.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], verbose=False)

    calib_tscv = TimeSeriesSplit(n_splits=3)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=calib_tscv)
    model.fit(X_calib, y_calib)
    logger.info("CalibratedClassifierCV (isotonic, cv=3-fold-ts) fitted on calib slice")

    # --- Threshold: auto-sweep on calib, or honour CFG override ---
    calib_prob = model.predict_proba(X_calib)[:, 1]
    if CFG.long_prob_threshold is None:
        best_threshold = _find_best_threshold(y_calib, calib_prob, label="XGB ")
    else:
        best_threshold = float(CFG.long_prob_threshold)
        logger.info(f"XGB using fixed threshold={best_threshold:.3f} (from CFG)")

    # --- Final test evaluation ---
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    logger.info(f"XGB calibrated  ROC-AUC={auc:.4f}  Avg-Precision={ap:.4f}")
    logger.info(
        f"XGB prob stats: min={y_prob.min():.3f} max={y_prob.max():.3f} "
        f"mean={y_prob.mean():.3f} >0.5: {(y_prob>0.5).sum()}"
    )

    y_pred = (y_prob >= best_threshold).astype(int)
    logger.info(f"XGB threshold={best_threshold:.3f}  predicted positives: {y_pred.sum()} / {len(y_pred)}")

    if y_pred.sum() == 0:
        softer = float(np.percentile(y_prob, 80))
        y_pred = (y_prob >= softer).astype(int)
        logger.warning(f"Recall(1)=0 at {best_threshold:.3f} — fallback to p80={softer:.3f}")
        best_threshold = softer

    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X_fit)
    shap_imp  = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 SHAP features:\n{shap_imp.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "xgb_model.pkl"))
    # Return best_threshold so main.py can pass it to generate_signal_from_prob
    return model, shap_imp, (X_test, y_test, y_prob), best_threshold
