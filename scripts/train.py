"""Model training pipeline for California housing price prediction.

Production-grade script featuring:
- Modular functional architecture (ETL, training, evaluation, persistence)
- Configurable CLI parameters & hyperparameters
- Structured logging with timestamps
- Data home / external dataset path support
- Evaluation metrics and metadata export (JSON)
- Post-training artifact verification / smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


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


def load_data(
    data_dir: Path | None = None,
    data_path: Path | None = None,
    target_col: str = "MedHouseVal",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and target either from a specified file or scikit-learn dataset."""
    if data_path is not None:
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        logger.info("Loading dataset from custom path: %s", data_path)
        if data_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format for dataset: {data_path.suffix}")

        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}"
            )
        features = df.drop(columns=[target_col])
        target = df[target_col]
    else:
        logger.info("Fetching California housing dataset (data_home=%s)...", data_dir)
        if data_dir is not None:
            data_dir.mkdir(parents=True, exist_ok=True)
        dataset = fetch_california_housing(
            data_home=str(data_dir) if data_dir else None,
            as_frame=True,
        )
        features = dataset.data
        target = dataset.target

    logger.info(
        "Successfully loaded dataset: %d samples, %d features.",
        features.shape[0],
        features.shape[1],
    )
    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets."""
    logger.info(
        "Splitting data (test_size=%.2f, random_state=%d)...",
        test_size,
        random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Split complete: train_samples=%d, test_samples=%d.",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Instantiate and train RandomForestRegressor."""
    logger.info(
        "Training RandomForestRegressor (n_estimators=%d, max_depth=%s, min_samples_split=%d, random_state=%d, n_jobs=%d)...",
        n_estimators,
        max_depth,
        min_samples_split,
        random_state,
        n_jobs,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate model on test set and return error metrics."""
    logger.info("Evaluating model on test split...")
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    logger.info(
        "Evaluation results -> MAE: %.4f | RMSE: %.4f | R2: %.4f", mae, rmse, r2
    )
    return metrics


def save_artifacts(
    model: RandomForestRegressor,
    metrics: dict[str, float],
    feature_names: list[str],
    hyperparameters: dict[str, object],
    model_path: Path,
    metrics_path: Path | None = None,
) -> None:
    """Save trained model pickle and metadata/metrics JSON."""
    from sklearn import __version__ as sklearn_version

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Serializing model artifact to %s...", model_path)
    joblib.dump(model, model_path)

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "sklearn_version": sklearn_version,
            "model_type": model.__class__.__name__,
            "hyperparameters": hyperparameters,
            "feature_names": feature_names,
            "metrics": metrics,
        }
        logger.info("Exporting metrics and metadata to %s...", metrics_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def verify_saved_model(model_path: Path, sample_features: pd.DataFrame) -> None:
    """Smoke test to verify that the saved model artifact can be reloaded and run predictions."""
    logger.info("Verifying saved model artifact by reloading from %s...", model_path)
    loaded_model: RandomForestRegressor = joblib.load(model_path)
    sample = sample_features.head(1)
    prediction = loaded_model.predict(sample)
    logger.info(
        "Verification successful. Sample prediction shape: %s, output value: %s",
        prediction.shape,
        prediction[0],
    )


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

        X_train, X_test, y_train, y_test = split_data(
            features=features,
            target=target,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        model = train_model(
            X_train=X_train,
            y_train=y_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

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
