import logging
from typing import Any

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
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
