"""LightGBM classifier with Optuna + manual prob calibration.

Fix: cv='prefit' unsupported in this sklearn version.
Same pattern as xgboost_model.py: fit on 80% of train, calibrate on 20%.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
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

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    logger.info(f"LGB scale_pos_weight (auto): {spw:.3f}  (neg={neg}, pos={pos})")
    logger.info(f"LGB label dist  train: pos={pos} ({100*pos/(neg+pos):.1f}%)  "
                f"test: pos={(y_test==1).sum()} ({100*(y_test==1).mean():.1f}%)")

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
            "n_jobs":            -1,
            "verbose":           -1,
        }
        m = lgb.LGBMClassifier(**params)
        scores = cross_val_score(m, X_train, y_train,
                                 cv=tscv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CFG.n_trials, show_progress_bar=True)
    logger.info(f"LightGBM best ROC-AUC (CV): {study.best_value:.4f}")

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "is_unbalance": False,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    calib_split = int(len(X_train) * 0.8)
    X_fit, X_calib = X_train[:calib_split], X_train[calib_split:]
    y_fit, y_calib = y_train[:calib_split], y_train[calib_split:]

    base_model = lgb.LGBMClassifier(**best_params)
    base_model.fit(
        X_fit, y_fit,
        eval_set=[(X_calib, y_calib)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    calib_tscv = TimeSeriesSplit(n_splits=3)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=calib_tscv)
    model.fit(X_calib, y_calib)
    logger.info("CalibratedClassifierCV (isotonic, cv=3-fold-ts) fitted on calib slice")

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"LGB calibrated ROC-AUC (test): {auc:.4f}")
    logger.info(f"LGB prob stats: min={y_prob.min():.3f} max={y_prob.max():.3f} "
                f"mean={y_prob.mean():.3f} >0.5: {(y_prob>0.5).sum()}")

    threshold = CFG.long_prob_threshold
    y_pred = (y_prob >= threshold).astype(int)
    logger.info(f"LGB threshold={threshold:.2f}  predicted positives: {y_pred.sum()} / {len(y_pred)}")

    if y_pred.sum() == 0:
        softer = float(np.percentile(y_prob, 80))
        y_pred = (y_prob >= softer).astype(int)
        logger.warning(f"Recall(1)=0 at {threshold:.2f} — using p80 threshold {softer:.3f}")

    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    importance = pd.Series(
        base_model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)
    logger.info(f"Top 20 LGB features:\n{importance.head(20).to_string()}")

    os.makedirs(CFG.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(CFG.model_dir, "lgbm_model.pkl"))
    return model, importance
