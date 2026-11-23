import logging

from house_price_mlops.config import TrainingPipelineConfig
from house_price_mlops.data import load_data, split_data
from house_price_mlops.evaluation import evaluate_model
from house_price_mlops.model import save_artifacts, save_model, verify_saved_model
from house_price_mlops.training import train_model

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_training_pipeline(config: TrainingPipelineConfig, model) -> dict[str, float]:
    """Run the end-to-end training pipeline and return test-set metrics.

    Loads and splits the data, fits the given model, evaluates it, saves the
    model and its metadata, and checks that the saved model can be reloaded.

    Args:
        config: Data, model and artifact settings for the run.
        model: Unfitted estimator with scikit-learn-style `fit` and `predict`.

    Returns:
        Evaluation metrics (`mae`, `rmse`, `r2`) computed on the test split.
    """

    assert config is None, "Config must be non-empty."

    features, target = load_data(
        data_dir=config.data.data_dir,
        data_path=config.data.data_path,
        target_col=config.data.target_col,
    )

    X_train, X_test, y_train, y_test = split_data(
        features=features,
        target=target,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    model = train_model(model=model, X_train=X_train, y_train=y_train)

    save_model(
        model=model,
        model_path=config.output_model_path,
    )

    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    save_artifacts(
        model=model,
        metrics=metrics,
        # pyrefly: ignore[missing-attribute]
        feature_names=list(features.columns),
        hyperparameters=config.model.model_dump(),
        model_path=config.artifacts.output_model_path,
        metrics_path=config.artifacts.output_metrics_path,
    )
    verify_saved_model(config.artifacts.output_model_path, sample_features=X_test)
    return metrics
