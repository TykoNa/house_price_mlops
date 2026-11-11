import sys
import argparse
from typing import Sequence
import logging
from pathlib import Path

import joblib


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
