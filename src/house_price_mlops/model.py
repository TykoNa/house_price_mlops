import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
from pandas import DataFrame

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="=Load model for California housing price prediction."
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/rfr.pkl"),
        help="Path where the trained model pickle will be saved (default: artifacts/rfr.pkl).",
    )
    return parser.parse_args(args)


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in {model_path}!")

    model = joblib.load(model_path)
    return model


def save_artifacts(
    model,
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


def verify_saved_model(model_path: Path, sample_features: DataFrame) -> None:
    """Smoke test to verify that the saved model artifact can be reloaded and run predictions."""
    logger.info("Verifying saved model artifact by reloading from %s...", model_path)
    loaded_model = load_model(model_path=model_path)
    sample = sample_features.head(1)
    prediction = loaded_model.predict(sample)
    logger.info(
        "Verification successful. Sample prediction shape: %s, output value: %s",
        prediction.shape,
        prediction[0],
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()

    try:
        load_model(model_path=args.model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception(f"Model loading failed with an error: \n {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
