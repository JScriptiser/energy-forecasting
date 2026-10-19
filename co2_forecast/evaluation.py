"""Rolling-origin backtest, error metrics and empirical prediction intervals."""

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from co2_forecast.config import HORIZON, LATENCY_HOURS, LOCAL_TZ, REFIT_EVERY_DAYS, SEED
from co2_forecast.features import add_derived, build_rows, local_midnights

# NED does not publish an archive of its own forecasts, so the backtest feeds the models actual
# renewable production. To see what forecast errors do, actuals can be turned into synthetic forecasts
# whose error grows with the lead day (1..7). The levels are assumptions in the range reported for
# aggregated national forecasts: wind - additive error, std as a share of installed capacity; solar and
# biomass - multiplicative, mean-preserving error with a relative std. Biomass follows dispatch rather
# than weather, so its error is assumed to grow too. Errors are autocorrelated over the hours of one
# forecast (AR(1)), onshore and offshore wind are correlated.
WIND_SIGMA = np.array([0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18])
SOLAR_SIGMA = np.array([0.15, 0.20, 0.25, 0.30, 0.33, 0.36, 0.40])
BIOMASS_SIGMA = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25])
WIND_RHO, SOLAR_RHO, BIOMASS_RHO, WIND_CORR = 0.97, 0.95, 0.99, 0.7


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """n independent unit-variance AR(1) paths over the forecast horizon, shape (n, HORIZON)."""
    shocks = rng.standard_normal((n, HORIZON))
    path = np.empty_like(shocks)
    path[:, 0] = shocks[:, 0]
    for h in range(1, HORIZON):
        path[:, h] = rho * path[:, h - 1] + np.sqrt(1 - rho**2) * shocks[:, h]
    return path


def with_forecast_errors(rows: pd.DataFrame, scale: float = 1.0, seed: int = SEED) -> pd.DataFrame:
    """Replace actual low-carbon production by synthetic forecasts (0 = perfect, 1 = assumed typical)."""
    if scale == 0:
        return rows
    rng = np.random.default_rng(seed)
    codes, origins = pd.factorize(rows["origin"])
    h = rows["h"].to_numpy()
    day = h // 24
    common = _ar1(rng, len(origins), WIND_RHO)
    wind_on = WIND_CORR * common + np.sqrt(1 - WIND_CORR**2) * _ar1(rng, len(origins), WIND_RHO)
    wind_off = WIND_CORR * common + np.sqrt(1 - WIND_CORR**2) * _ar1(rng, len(origins), WIND_RHO)
    solar = _ar1(rng, len(origins), SOLAR_RHO)
    biomass = _ar1(rng, len(origins), BIOMASS_RHO)

    out = rows.copy()
    year = out["time"].dt.year
    for column, noise in (("wind_onshore", wind_on), ("wind_offshore", wind_off)):
        capacity = out.groupby(year)[column].transform(lambda s: s.quantile(0.995))
        error = scale * WIND_SIGMA[day] * capacity * noise[codes, h]
        out[column] = (out[column] + error).clip(lower=0, upper=capacity)
    for column, sigma, noise in (("solar", SOLAR_SIGMA, solar), ("biomass", BIOMASS_SIGMA, biomass)):
        s = scale * sigma[day]
        out[column] = out[column] * np.exp(s * noise[codes, h] - s**2 / 2)  # lognormal with mean 1
    return add_derived(out)


def last_complete_origin(grid: pd.DataFrame) -> pd.Timestamp:
    """Latest local midnight whose whole 168-hour window has actuals."""
    last_actual = grid["ef"].last_valid_index()
    candidates = local_midnights(str((last_actual - pd.Timedelta(days=9)).date()), str(last_actual.date()))
    return candidates[candidates + pd.Timedelta(hours=HORIZON - 1) <= last_actual][-1]


def backtest(
    grid: pd.DataFrame,
    factories: dict[str, Callable[[], object]],
    train_start: str,
    test_start: str,
    test_end: pd.Timestamp | None = None,
    refit_every_days: int = REFIT_EVERY_DAYS,
    train_stride_days: int = 1,
    train_window_days: int | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    seed: int = SEED,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Simulate forecasts issued every local midnight from test_start to test_end.

    Models are refitted every `refit_every_days` on the rows whose target hour was already visible
    at that moment (time < refit origin - latency), so no future information leaks into training.
    `train_noise` / `test_noise` replace actual renewables by synthetic forecasts (see with_forecast_errors).
    Returns one row per (origin, hour ahead) with the actual value `y` and one column per model.
    """
    test_end = test_end if test_end is not None else last_complete_origin(grid)
    origins = local_midnights(train_start, str(test_end.tz_convert(LOCAL_TZ).date()))
    rows = build_rows(grid, origins)
    train_rows = with_forecast_errors(rows, train_noise, seed)
    test_rows = with_forecast_errors(rows, test_noise, seed + 1)
    is_train_origin = (rows["origin"] - rows["origin"].min()).dt.days % train_stride_days == 0

    test_origins = origins[origins >= pd.Timestamp(test_start, tz=LOCAL_TZ)]
    blocks = [test_origins[i : i + refit_every_days] for i in range(0, len(test_origins), refit_every_days)]
    results = []
    for block in blocks:
        refit_at = block[0]
        visible = rows["time"] < refit_at - pd.Timedelta(hours=LATENCY_HOURS)
        keep = (rows["origin"] < refit_at) & visible & rows["y"].notna() & is_train_origin
        if train_window_days:
            keep &= rows["origin"] >= refit_at - pd.Timedelta(days=train_window_days)
        train = train_rows[keep]
        test = test_rows[rows["origin"].isin(block)].copy()
        started = time.perf_counter()
        for name, factory in factories.items():
            test[name] = factory().fit(train).predict(test)
        log(
            f"  block {refit_at.tz_convert('Europe/Amsterdam'):%Y-%m-%d}: {len(train):,} train rows, "
            f"{len(block)} origins, {time.perf_counter() - started:.1f}s"
        )
        results.append(test)
    out = pd.concat(results, ignore_index=True)
    return out[out["y"].notna()]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - y
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "WAPE_%": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(y))),
        "bias": float(np.mean(error)),
    }


def summarize(preds: pd.DataFrame, models: list[str], baseline: str) -> pd.DataFrame:
    """Overall metrics per model; skill = MAE reduction relative to the baseline."""
    table = pd.DataFrame({m: metrics(preds["y"].to_numpy(), preds[m].to_numpy()) for m in models}).T
    table["skill_vs_" + baseline + "_%"] = 100 * (1 - table["MAE"] / table.loc[baseline, "MAE"])
    return table.sort_values("MAE")


def mae_by(preds: pd.DataFrame, models: list[str], key: pd.Series) -> pd.DataFrame:
    """MAE of every model grouped by `key` (lead day, local hour, month...)."""
    errors = preds[models].sub(preds["y"], axis=0).abs()
    return errors.groupby(key.to_numpy()).mean()


def lead_day(preds: pd.DataFrame) -> pd.Series:
    return (preds["h"] // 24 + 1).rename("lead_day")


def residual_quantiles(preds: pd.DataFrame, model: str, levels=(0.1, 0.9)) -> pd.DataFrame:
    """Empirical 10% / 90% quantiles of (actual - forecast) for every hour ahead: an 80% interval.

    Origins are local midnights, so the hour ahead also fixes the hour of day; the quantiles therefore
    follow both the growth of the error with the horizon and its daily shape. To reduce noise each value
    is averaged with the same hour of the neighbouring lead days.
    """
    residual = preds["y"] - preds[model]
    raw = residual.groupby(preds["h"].to_numpy()).quantile(list(levels)).unstack()
    raw = raw.reindex(range(HORIZON)).interpolate(limit_direction="both")
    smooth = {}
    for name, column in zip(["lower", "upper"], raw.columns, strict=True):
        by_day = raw[column].to_numpy().reshape(HORIZON // 24, 24)
        smooth[name] = pd.DataFrame(by_day).rolling(3, center=True, min_periods=1).mean().to_numpy().ravel()
    return pd.DataFrame(smooth, index=pd.RangeIndex(HORIZON, name="h"))


def interval_coverage(preds: pd.DataFrame, model: str, quantiles: pd.DataFrame) -> float:
    """Share of actuals inside forecast + residual quantiles (quantiles from another period)."""
    h = preds["h"].to_numpy()
    lower = preds[model].to_numpy() + quantiles["lower"].reindex(h).to_numpy()
    upper = preds[model].to_numpy() + quantiles["upper"].reindex(h).to_numpy()
    return float(np.mean((preds["y"] >= lower) & (preds["y"] <= upper)))
