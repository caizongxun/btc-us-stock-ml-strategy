"""XGBoost classifier with Optuna hyperparameter search + SHAP explanation.

Key fix: scale_pos_weight is derived from training data automatically,
not left as a random search parameter — ensures the model always sees
a balanced signal regardless of label distribution.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
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

    # Auto scale_pos_weight from training set
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    logger.info(f"XGB scale_pos_weight (auto): {spw:.3f}  (neg={neg}, pos={pos})")

    tscv = TimeSeriesSplit(n_splits=CFG.cv_splits)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight":  spw,   # fixed — not a search parameter
            "eval_metric":       "logloss",
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="f1", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": spw,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    })
    logger.info(f"XGBoost best F1 (CV): {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    logger.info("\n" + classification_report(y_test, y_pred))

    # SHAP feature importance
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    importance  = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 SHAP features:\n{importance.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "xgboost_model.pkl"))
    importance.to_csv(os.path.join(CFG.model_dir, "shap_importance.csv"))

    return model, importance, (X_test, y_test, y_prob)
