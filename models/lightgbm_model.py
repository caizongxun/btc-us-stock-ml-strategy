"""LightGBM classifier with Optuna search — faster alternative to XGBoost.

Key fix: scale_pos_weight derived from training data automatically.
Adds is_unbalance=False + explicit scale_pos_weight so LGB doesn't
double-compensate.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report
from loguru import logger
import joblib, os
from config import CFG

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_lightgbm(df: pd.DataFrame, label_col: str = "y_binary"):
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
    logger.info(f"LGB scale_pos_weight (auto): {spw:.3f}  (neg={neg}, pos={pos})")

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
            "scale_pos_weight":  spw,   # fixed — not a search parameter
            "is_unbalance":      False,  # don't double-compensate
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="f1", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "is_unbalance": False,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    logger.info(f"LightGBM best F1 (CV): {study.best_value:.4f}")

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred))

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 LGB features:\n{importance.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "lgbm_model.pkl"))
    return model, importance
