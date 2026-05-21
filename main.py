"""Full pipeline runner"""
import argparse
import json
import os
from loguru import logger
from config import CFG

def parse_args():
    p = argparse.ArgumentParser(description="BTC + US Stock ML Strategy Pipeline")
    p.add_argument("--asset", default="BTC", choices=["BTC","SPY","QQQ"])
    p.add_argument("--target_days", type=int, default=3)
    p.add_argument("--mode", default="full",
                   choices=["full","rules_only","regime_only","export","backtest"])
    p.add_argument("--long_thresh", type=float, default=0.015)
    p.add_argument("--short_thresh", type=float, default=-0.015)
    return p.parse_args()

def main():
    args = parse_args()
    CFG.target_asset = args.asset
    CFG.target_days = args.target_days
    CFG.long_threshold = args.long_thresh
    CFG.short_threshold = abs(args.short_thresh)

    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.model_dir, exist_ok=True)

    # Step 1: Build features
    logger.info("=" * 60)
    logger.info(f"Pipeline: asset={args.asset}, target={args.target_days}d, mode={args.mode}")
    logger.info("=" * 60)

    from features.build_features import build_feature_matrix
    df = build_feature_matrix(target_asset=args.asset, target_days=args.target_days)

    feature_cols = [c for c in df.columns if c not in ["y","y_binary"]]
    logger.info(f"Total features: {len(feature_cols)}")

    if args.mode in ["full", "regime_only"]:
        # Regime detection already done inside build_feature_matrix
        logger.info("Regime detection: done (embedded in features)")

    if args.mode in ["full", "rules_only"]:
        # Step 2: Train XGBoost
        logger.info("Training XGBoost...")
        from models.xgboost_model import train_xgboost
        xgb_model, shap_imp, (X_test, y_test, y_prob) = train_xgboost(df)

        # Step 3: Train LightGBM
        logger.info("Training LightGBM...")
        from models.lightgbm_model import train_lightgbm
        lgb_model, lgb_imp = train_lightgbm(df)

        # Step 4: Mine binary rules from top SHAP features
        logger.info("Mining binary rules...")
        from models.binary_rule_miner import mine_binary_rules
        top_features = list(shap_imp.index[:30])
        rule_result = mine_binary_rules(df, top_features)

        # Step 5: Generate signals
        from strategy.signal_generator import generate_signal_from_prob
        import numpy as np, pandas as pd
        signal_prob = generate_signal_from_prob(y_prob)

        # Step 6: Multi-signal
        test_df = df.iloc[int(len(df) * (1 - CFG.test_size)):].copy()
        signal_series = pd.Series(signal_prob, index=test_df.index, name="signal")
        from strategy.multi_signal import build_multi_signal
        multi_sig, sig_components = build_multi_signal(test_df, signal_series)

        # Step 7: Backtest
        from data.fetch_btc import fetch_btc_ohlcv
        from data.fetch_stocks import fetch_stocks
        from backtest.evaluate import evaluate_signal

        if args.asset == "BTC":
            price_full = fetch_btc_ohlcv()["close"]
        else:
            price_full = fetch_stocks([args.asset])[args.asset]["close"]
        price_test = price_full.reindex(test_df.index)

        metrics_ml = evaluate_signal(price_test, signal_series, label="xgb_ml")
        metrics_multi = evaluate_signal(price_test, multi_sig, label="multi_signal")
        logger.info(f"XGB Signal Metrics: {metrics_ml}")
        logger.info(f"Multi Signal Metrics: {metrics_multi}")

        # Step 8: Export QuantDingers strategy
        from strategy.quantdingers_export import export_quantdingers_strategy
        export_quantdingers_strategy(
            thresholds=rule_result["thresholds"],
            top_features=top_features,
            shap_importance=shap_imp,
            test_f1=rule_result["test_f1"],
        )

        # Save summary
        summary = {
            "xgb_metrics": metrics_ml,
            "multi_signal_metrics": metrics_multi,
            "top10_features": list(shap_imp.head(10).index),
            "rule_f1": rule_result["test_f1"],
            "rule_thresholds": rule_result["thresholds"],
        }
        with open(os.path.join(CFG.output_dir, "pipeline_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.success("Pipeline complete! Check ./outputs/ for results.")

if __name__ == "__main__":
    main()
