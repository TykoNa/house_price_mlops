import json

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from house_price_mlops.model import (
    load_model,
    main,
    parse_args,
    save_artifacts,
    save_model,
    verify_saved_model,
)


@pytest.fixture
def fitted_model():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    return LinearRegression().fit(X, [2.0, 4.0, 6.0]), X


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_model(tmp_path / "missing.pkl")


def test_load_model_corrupted_file_raises(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="Failed to deserialize model"):
        load_model(path)


def test_save_model_creates_nested_parent_dirs(tmp_path, fitted_model):
    model, X = fitted_model
    path = tmp_path / "a" / "b" / "c" / "model.pkl"

    save_model(model, path)

    assert path.exists()
    assert load_model(path).predict(X).tolist() == pytest.approx([2.0, 4.0, 6.0])


def test_save_artifacts_without_metrics_path_writes_nothing(tmp_path, fitted_model):
    model, _ = fitted_model

    save_artifacts(model, {}, [], {}, tmp_path / "model.pkl", metrics_path=None)

    assert list(tmp_path.iterdir()) == []


def test_save_artifacts_empty_inputs_and_nested_metrics_dir(tmp_path, fitted_model):
    model, _ = fitted_model
    metrics_path = tmp_path / "x" / "y" / "metrics.json"

    save_artifacts(model, {}, [], {}, tmp_path / "model.pkl", metrics_path)

    data = json.loads(metrics_path.read_text())
    assert data["metrics"] == {}
    assert data["feature_names"] == []
    assert data["hyperparameters"] == {}
    assert data["model_type"] == "LinearRegression"


def test_verify_saved_model_empty_dataframe_raises(tmp_path, fitted_model):
    model, X = fitted_model
    path = tmp_path / "model.pkl"
    save_model(model, path)

    with pytest.raises(ValueError):
        verify_saved_model(path, X.iloc[0:0])


def test_verify_saved_model_single_row_ok(tmp_path, fitted_model):
    model, X = fitted_model
    path = tmp_path / "model.pkl"
    save_model(model, path)

    verify_saved_model(path, X.iloc[:1])


def test_parse_args_default_model_path():
    assert str(parse_args([]).model_path) == "artifacts/rfr.pkl"


def test_main_missing_model_exits_with_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--model-path", str(tmp_path / "missing.pkl")])
    assert exc.value.code == 1
