"""XGBoost classifier with Optuna hyperparameter search + SHAP feature importance.

Fixes:
- scale_pos_weight auto-calculated from training labels
- Returns (model, shap_importance, (X_test, y_test, y_prob)) for downstream use
- Soft threshold fallback if Recall(1)=0
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report
from loguru import logger
import joblib, os
from config import CFG

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_xgboost(df: pd.DataFrame, label_col: str = "y_binary"):
    feature_cols = [c for c in df.columns if c not in ["y", "y_binary"]]
    X = df[feature_cols].values
    y = df[label_col].values

    split_idx = int(len(X) * (1 - CFG.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Auto scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    logger.info(f"XGB scale_pos_weight (auto): {spw:.3f}  (neg={neg}, pos={pos})")
    logger.info(f"XGB label distribution  train: pos={pos} ({100*pos/(neg+pos):.1f}%)  "
                f"test: pos={(y_test==1).sum()} ({100*(y_test==1).mean():.1f}%)")

    tscv = TimeSeriesSplit(n_splits=CFG.cv_splits)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 30),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  spw,
            "eval_metric":       "logloss",
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="f1", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    logger.info(f"XGBoost best F1 (CV): {study.best_value:.4f}")

    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Prob threshold (not argmax) ---
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = CFG.long_prob_threshold
    y_pred = (y_prob >= threshold).astype(int)
    logger.info(f"XGB threshold={threshold:.2f}  predicted positives: {y_pred.sum()} / {len(y_pred)}")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    # Soft-adjust threshold if recall(1) is still 0
    if y_pred.sum() == 0:
        softer = max(0.45, threshold - 0.10)
        y_pred = (y_prob >= softer).astype(int)
        logger.warning(f"Recall(1)=0 at threshold {threshold:.2f} — retrying at {softer:.2f}")
        logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    # SHAP importance
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)
    shap_imp = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 SHAP features:\n{shap_imp.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "xgb_model.pkl"))
    return model, shap_imp, (X_test, y_test, y_prob)
