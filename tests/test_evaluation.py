from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from co2_forecast.config import HORIZON, LATENCY_HOURS
from co2_forecast.evaluation import (
    backtest,
    interval_coverage,
    metrics,
    residual_quantiles,
    with_forecast_errors,
)
from co2_forecast.features import build_rows, local_midnights


class Spy:
    """Records what the backtest lets the model see."""

    seen: ClassVar[list] = []

    def fit(self, rows):
        Spy.seen.append(rows["time"].max())
        return self

    def predict(self, rows):
        Spy.seen.append(("first_origin", rows["origin"].min()))
        return np.zeros(len(rows))


def test_backtest_never_trains_on_hours_hidden_at_the_refit(grid):
    Spy.seen = []
    backtest(grid, {"spy": Spy}, "2025-02-01", "2025-03-01", refit_every_days=7, log=lambda _: None)
    pairs = list(zip(Spy.seen[::2], Spy.seen[1::2], strict=True))
    assert len(pairs) >= 2
    for last_train_hour, (_, first_origin) in pairs:
        assert last_train_hour < first_origin - pd.Timedelta(hours=LATENCY_HOURS)


def test_backtest_returns_every_hour_of_every_complete_window(grid):
    preds = backtest(grid, {"spy": Spy}, "2025-02-01", "2025-03-01", refit_every_days=7, log=lambda _: None)
    assert preds.groupby("origin").size().eq(HORIZON).all()
    assert preds["y"].notna().all()


def test_forecast_errors_are_off_at_scale_zero_and_physical_otherwise(grid):
    rows = build_rows(grid, local_midnights("2025-02-10", "2025-02-20"))
    assert with_forecast_errors(rows, 0) is rows
    noisy = with_forecast_errors(rows, 1.0)
    assert (noisy.loc[rows["solar"] == 0, "solar"] == 0).all()  # no sun at night, also in a forecast
    assert (noisy[["wind_onshore", "wind_offshore", "biomass"]] >= 0).all().all()
    assert not np.allclose(noisy["wind_onshore"], rows["wind_onshore"])
    assert noisy["low_carbon"].equals(
        noisy[["solar", "wind_onshore", "wind_offshore", "biomass"]].sum(axis=1)
    )
    # the error grows with the lead day
    gap = (noisy["wind_onshore"] - rows["wind_onshore"]).abs().groupby(rows["h"] // 24).mean()
    assert gap.iloc[-1] > gap.iloc[0]


def test_metrics_on_a_hand_checked_example():
    result = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.0]))
    assert result["MAE"] == pytest.approx(0.5)
    assert result["RMSE"] == pytest.approx(np.sqrt((0.25 + 1) / 3))
    assert result["bias"] == pytest.approx(-1 / 6)
    assert result["WAPE_%"] == pytest.approx(100 * 1.5 / 6)


def test_interval_from_residual_quantiles_covers_about_80_percent():
    rng = np.random.default_rng(1)
    h = np.tile(np.arange(HORIZON), 400)
    preds = pd.DataFrame({"h": h, "model": 0.0, "y": rng.normal(0, 1, len(h))})
    bands = residual_quantiles(preds, "model")
    assert bands.shape == (HORIZON, 2) and (bands["lower"] < bands["upper"]).all()
    assert interval_coverage(preds, "model", bands) == pytest.approx(0.8, abs=0.03)
