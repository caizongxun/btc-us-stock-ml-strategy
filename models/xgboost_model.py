"""XGBoost classifier with Optuna + SHAP + CalibratedClassifierCV.

Same calibration fix as lightgbm_model.py:
  Optuna CV metric -> roc_auc
  Final model wrapped with CalibratedClassifierCV (isotonic)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
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
    logger.info(f"XGB label dist  train: pos={pos} ({100*pos/(neg+pos):.1f}%)  "
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
        # roc_auc: threshold-free, optimises probability separation
        scores = cross_val_score(model, X_train, y_train,
                                 cv=tscv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)
    logger.info(f"XGBoost best ROC-AUC (CV): {study.best_value:.4f}")

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    base_model = xgb.XGBClassifier(**best_params)
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Calibrate probabilities
    model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    model.fit(X_train, y_train)
    logger.info("CalibratedClassifierCV (isotonic) fitted")

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"XGB calibrated ROC-AUC (test): {auc:.4f}")
    logger.info(f"XGB prob stats: min={y_prob.min():.3f} max={y_prob.max():.3f} "
                f"mean={y_prob.mean():.3f} >0.5: {(y_prob>0.5).sum()}")

    threshold = CFG.long_prob_threshold
    y_pred = (y_prob >= threshold).astype(int)
    logger.info(f"XGB threshold={threshold:.2f}  predicted positives: {y_pred.sum()} / {len(y_pred)}")

    if y_pred.sum() == 0:
        softer = float(np.percentile(y_prob, 80))
        y_pred = (y_prob >= softer).astype(int)
        logger.warning(f"Recall(1)=0 at {threshold:.2f} — using percentile-80 threshold {softer:.3f}")

    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    # SHAP on base model (before calibration wrapper)
    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X_train)
    shap_imp = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 SHAP features:\n{shap_imp.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "xgb_model.pkl"))
    return model, shap_imp, (X_test, y_test, y_prob)
