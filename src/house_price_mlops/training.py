from typing import Any


def train_model(model: Any, X_train: Any, y_train: Any) -> Any:
    """Fit a machine learning estimator on training data.
    Validates that the provided model implements a callable `fit` method
    conforming to the scikit-learn estimator interface before executing training.
    Args:
        model: Estimator instance or pipeline implementing a callable `.fit()` method.
        X_train: Training feature matrix (pandas DataFrame, numpy array, or sparse matrix).
        y_train: Target values corresponding to `X_train` (pandas Series or numpy array).
    Returns:
        The fitted model instance returned by `model.fit(X_train, y_train)`.
    Raises:
        AttributeError: If `model` does not have a callable `fit` method.
    """
    fit_method = getattr(model, "fit", None)
    if callable(fit_method):
        return model.fit(X_train, y_train)

    model_name = getattr(model, "__name__", type(model).__name__)
    raise AttributeError(f"'{model_name}' does not have a callable 'fit' method!")
