import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def split_data(
    features: Any,
    target: Any,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Any, Any, Any, Any]:
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


def load_data(
    data_dir: Path | None = None,
    data_path: Path | None = None,
    target_col: str = "MedHouseVal",
) -> tuple[pd.DataFrame | None, pd.Series | None]:
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
