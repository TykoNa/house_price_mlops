import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from house_price_mlops.training import train_model

from .utils import BadFit, NoFit  # type: ignore


def test_train_model_without_fit_method_raises():
    """Border: object has no `fit` attribute at all."""

    X, y = np.array([[1.0], [2.0]]), np.array([1.0, 2.0])

    with pytest.raises(AttributeError, match="NoFit.*does not have a callable 'fit'"):
        train_model(NoFit(), X, y)


def test_train_model_non_callable_fit_raises():
    """Border: `fit` exists but is not callable."""

    X, y = np.array([[1.0], [2.0]]), np.array([1.0, 2.0])

    with pytest.raises(AttributeError, match="does not have a callable 'fit'"):
        train_model(BadFit(), X, y)


def test_train_model_single_sample_fits():
    """Border: minimal dataset with a single row still trains and returns the model."""
    X, y = np.array([[5.0]]), np.array([8.0])
    model = LinearRegression()

    trained = train_model(model, X, y)

    assert trained is model
    check_is_fitted(trained)
    assert trained.predict(X)[0] == pytest.approx(8.0)
