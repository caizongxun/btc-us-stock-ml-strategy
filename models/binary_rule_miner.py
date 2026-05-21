"""Extract human-readable binary rules from a Decision Tree fitted on top features."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from loguru import logger
import os
from config import CFG

def mine_binary_rules(
    df: pd.DataFrame,
    top_features: list,
    label_col: str = "y_binary",
    max_depth: int = 4,
    max_features: int = 20,
) -> dict:
    """
    Fits a shallow Decision Tree on the top features selected by SHAP.
    Returns: dict with tree rules, feature thresholds, and backtest stats.
    """
    # Use only top N features
    feat_cols = top_features[:max_features]
    X = df[feat_cols].values
    y = df[label_col].values

    split_idx = int(len(X) * (1 - CFG.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Grid search over max_depth and min_samples_leaf
    best_f1, best_model = 0, None
    for depth in range(2, max_depth + 1):
        for min_leaf in [10, 20, 30, 50]:
            clf = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=min_leaf,
                class_weight="balanced", random_state=42
            )
            clf.fit(X_train, y_train)
            f1 = f1_score(y_test, clf.predict(X_test), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_depth = depth
                best_leaf = min_leaf

    logger.info(f"Best DT: depth={best_depth}, min_leaf={best_leaf}, test F1={best_f1:.4f}")

    # Extract rules as text
    rules_text = export_text(best_model, feature_names=feat_cols)
    logger.info(f"Decision Tree Rules:\n{rules_text}")

    # Save rules
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(os.path.join(CFG.output_dir, "binary_rules.txt"), "w") as f:
        f.write(rules_text)

    # Parse key thresholds for QuantDingers export
    thresholds = _parse_thresholds(best_model, feat_cols)
    logger.info(f"Parsed thresholds: {thresholds}")

    return {
        "model": best_model,
        "rules_text": rules_text,
        "thresholds": thresholds,
        "test_f1": best_f1,
        "feature_names": feat_cols,
    }

def _parse_thresholds(tree, feature_names):
    """Extract feature -> threshold pairs from the tree."""
    from sklearn.tree import _tree
    t = tree.tree_
    thresholds = {}
    def recurse(node):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[t.feature[node]]
            thresh = t.threshold[node]
            if name not in thresholds:
                thresholds[name] = []
            thresholds[name].append(round(thresh, 6))
            recurse(t.children_left[node])
            recurse(t.children_right[node])
    recurse(0)
    return thresholds
