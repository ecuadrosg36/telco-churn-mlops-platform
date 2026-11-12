"""Tests for config helpers."""

import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import get_model_params


@pytest.mark.unit
@pytest.mark.parametrize("model_type", ["baseline", "advanced"])
def test_model_params_leave_out_the_type_label(model_type: str) -> None:
    params = get_model_params(model_type)

    assert "type" not in params
    assert params["random_state"] == 42


@pytest.mark.unit
def test_baseline_params_build_a_random_forest() -> None:
    params = get_model_params("baseline")

    model = RandomForestClassifier(**params)

    assert model.n_estimators == params["n_estimators"]
