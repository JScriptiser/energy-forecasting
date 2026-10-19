"""Forecasting models behind one interface: fit(rows) / predict(rows) on the frames from features.build_rows.

To add a model: write a class with `name`, `fit` and `predict`, and register a factory in MODELS.
The backtest and the final forecast pick it up from there; tests/test_models.py checks the contract.
"""

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from co2_forecast.config import SEED
from co2_forecast.features import ALL_FEATURES


class SeasonalNaive:
    """Same hour one week earlier (two weeks for the last day of the horizon, because of the data delay)."""

    name = "seasonal_naive"

    def fit(self, rows: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["ef_snaive"].fillna(rows["ef_last168"]).to_numpy()


class WeeklyProfile:
    """Mean of the same hour of the week over the last four visible weeks."""

    name = "weekly_profile"

    def fit(self, rows: pd.DataFrame) -> "WeeklyProfile":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["ef_profile"].fillna(rows["ef_last168"]).to_numpy()


class Physics:
    """No learning: CO2 factor = recent intensity of the rest of the mix * (1 - forecasted low-carbon share).

    The share uses the forecasted low-carbon production and the typical total production at that hour.
    It shows how much of the problem the accounting identity alone explains.
    """

    name = "physics"

    def fit(self, rows: pd.DataFrame) -> "Physics":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["ef_physics"].fillna(rows["ef_last168"]).to_numpy()


class RidgeModel:
    """Linear baseline: the physics feature and history levels plus one-hot hour of day and day type."""

    name = "ridge"
    categorical = ("hour", "day_type")
    numeric = (
        "h", "doy_sin", "doy_cos", "solar", "wind_onshore", "wind_offshore", "biomass", "low_carbon",
        "low_carbon_share", "ef_physics", "ef_snaive", "ef_profile", "ef_last24", "ef_last168",
        "total_profile", "intensity",
    )  # fmt: skip

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.columns = [*self.categorical, *self.numeric]

    def fit(self, rows: pd.DataFrame) -> "RidgeModel":
        encode = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), list(self.categorical)),
                (
                    "num",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    list(self.numeric),
                ),
            ]
        )
        self.model = make_pipeline(encode, Ridge(alpha=self.alpha))
        self.model.fit(rows[self.columns], rows["y"])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[self.columns])


class LightGBMModel:
    """Gradient boosting on all features; NaNs are handled natively.

    With `residual=True` the trees learn the correction to the physics estimate instead of the CO2 factor
    itself, so the model can follow a drift of the mix beyond the range seen in training.
    """

    def __init__(self, residual: bool = True, features: list[str] | None = None, **params):
        self.residual = residual
        self.features = features or ALL_FEATURES
        self.params = {
            "objective": "l2",
            "learning_rate": 0.05,
            "n_estimators": 600,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": 4,
            "random_state": SEED,
            "verbose": -1,
        } | params
        self.name = "lightgbm"

    def _base(self, rows: pd.DataFrame) -> np.ndarray:
        if not self.residual:
            return np.zeros(len(rows))
        return rows["ef_physics"].fillna(rows["ef_last168"]).fillna(rows["ef_last168"].mean()).to_numpy()

    def fit(self, rows: pd.DataFrame) -> "LightGBMModel":
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(rows[self.features], rows["y"].to_numpy() - self._base(rows))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self._base(rows) + self.model.predict(rows[self.features])


MODELS: dict[str, Callable[[], object]] = {
    "seasonal_naive": SeasonalNaive,
    "weekly_profile": WeeklyProfile,
    "physics": Physics,
    "ridge": RidgeModel,
    "lightgbm": LightGBMModel,
}

BASELINE = "seasonal_naive"
