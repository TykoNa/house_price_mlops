"""Model training pipeline for California housing price prediction.

Production-grade script orchestrating:
- Data extraction & splitting (house_price_mlops.data)
- Model fitting (house_price_mlops.training)
- Evaluation & metrics calculation (house_price_mlops.evaluation)
- Model persistence & smoke test verification (house_price_mlops.model)
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from house_price_mlops.model import save_artifacts, verify_saved_model

# Ensure src/ is on sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import RandomForestRegressor

from house_price_mlops.data import load_data, split_data
from house_price_mlops.evaluation import evaluate_model
from house_price_mlops.model import load_model
from house_price_mlops.training import train_model

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a RandomForestRegressor model for California housing price prediction."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to cache or locate fetched dataset files (default: data).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional path to custom CSV/parquet dataset. If omitted, uses fetch_california_housing.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="MedHouseVal",
        help="Target column name when loading custom dataset (default: MedHouseVal).",
    )
    parser.add_argument(
        "--output-model-path",
        type=Path,
        default=Path("artifacts/rfr.pkl"),
        help="Path where the trained model pickle will be saved (default: artifacts/rfr.pkl).",
    )
    parser.add_argument(
        "--output-metrics-path",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Path where evaluation metrics JSON will be saved (default: artifacts/metrics.json).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to allocate to test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splitting and model reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees in the forest (default: None).",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node (default: 2).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run during model training (default: -1).",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Main orchestration entry point."""
    setup_logging()
    args = parse_args(argv)

    try:
        features, target = load_data(
            data_dir=args.data_dir,
            data_path=args.data_path,
            target_col=args.target_col,
        )

        if features is None or target is None:
            raise ValueError("Dataset loading failed: features or target is None.")

        X_train, X_test, y_train, y_test = split_data(
            features=features,
            target=target,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        logger.info(
            "Training %s (n_estimators=%d, max_depth=%s, min_samples_split=%d, random_state=%d, n_jobs=%d)...",
            model.__class__.__name__,
            args.n_estimators,
            args.max_depth,
            args.min_samples_split,
            args.random_state,
            args.n_jobs,
        )
        model = train_model(model=model, X_train=X_train, y_train=y_train)
        logger.info("Model training completed.")

        metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

        hyperparams = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "random_state": args.random_state,
            "test_size": args.test_size,
        }

        save_artifacts(
            model=model,
            metrics=metrics,
            feature_names=list(features.columns),
            hyperparameters=hyperparams,
            model_path=args.output_model_path,
            metrics_path=args.output_metrics_path,
        )

        verify_saved_model(model_path=args.output_model_path, sample_features=X_test)
        logger.info("Training pipeline finished successfully.")
    except Exception:
        logger.exception("Training pipeline encountered a fatal error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
