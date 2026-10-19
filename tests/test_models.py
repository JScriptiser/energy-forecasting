import numpy as np
import pytest

from co2_forecast.features import build_rows, local_midnights
from co2_forecast.models import MODELS


@pytest.fixture
def panel(grid):
    rows = build_rows(grid, local_midnights("2025-02-10", "2025-03-10"))
    rows = rows[rows["y"].notna()]
    return rows[rows["origin"] < rows["origin"].max()], rows[rows["origin"] == rows["origin"].max()]


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_keeps_the_contract(name, panel):
    """fit(rows) returns the model, predict(rows) returns one finite value per row."""
    train, test = panel
    model = MODELS[name]()
    assert model.fit(train) is model
    prediction = model.predict(test)
    assert prediction.shape == (len(test),)
    assert np.isfinite(prediction).all()


def test_learned_models_beat_the_seasonal_naive_on_physics_like_data(panel):
    train, test = panel
    error = {name: np.mean(np.abs(MODELS[name]().fit(train).predict(test) - test["y"])) for name in MODELS}
    assert error["lightgbm"] < error["seasonal_naive"]
    assert error["ridge"] < error["seasonal_naive"]
