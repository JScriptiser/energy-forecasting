"""Features for direct multi-horizon forecasting.

A forecast is issued at an origin t0 (local midnight) for the hours t0 .. t0 + 167h.
Every (origin, target hour) pair is one row; its features come from three places:

1. The target hour itself: calendar (local hour, day type, season) and the expected low-carbon
   production at that hour. During training these are NED actuals; in production they are
   NED's own 7-day forecasts, which is exactly the information the assignment asks to use.
2. The history visible at the origin. NED data arrives with a delay, so only hours before
   t0 - LATENCY_HOURS can be used: the seasonal-naive value, recent levels of the CO2 factor,
   the typical total production at that hour of the week and the CO2 intensity of the
   non-low-carbon part of the mix.
3. The horizon h = t - t0, which lets the model decide how much to trust the history.

All lookups are index arithmetic on a gap-free hourly grid; the leakage test in
tests/test_features.py checks that nothing at or after the cutoff can change a feature.
"""

import warnings

import holidays
import numpy as np
import pandas as pd

from co2_forecast.config import HORIZON, LATENCY_HOURS, LOCAL_TZ, LOW_CARBON

WEEK = 168
PROFILE_WEEKS = 4
PAD = 8 * WEEK  # look-back room before the first hour of the grid

CALENDAR = ["hour", "dow", "day_type", "doy_sin", "doy_cos"]
EXOGENOUS = [*LOW_CARBON, "low_carbon"]
HISTORY = ["ef_snaive", "ef_profile", "ef_last24", "ef_last168", "total_profile", "intensity"]
DERIVED = ["low_carbon_share", "ef_physics"]
ALL_FEATURES = ["h", *CALENDAR, *EXOGENOUS, *HISTORY, *DERIVED]


def add_low_carbon(grid: pd.DataFrame) -> pd.DataFrame:
    """Sum of solar, onshore and offshore wind and biomass (all carry a zero CO2 factor in NED)."""
    grid = grid.copy()
    grid["low_carbon"] = grid[list(LOW_CARBON)].sum(axis=1, min_count=len(LOW_CARBON))
    return grid


def add_derived(rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute the features derived from the low-carbon inputs (in place, also returns the frame).

    Accounting identity: CO2 factor = intensity of the non-low-carbon mix * its share of total production.
    Total production at the target hour is unknown, so its typical value for that hour of the week is used.
    """
    rows["low_carbon"] = rows[list(LOW_CARBON)].sum(axis=1, min_count=len(LOW_CARBON))
    rows["low_carbon_share"] = rows["low_carbon"] / rows["total_profile"]
    rows["ef_physics"] = rows["intensity"] * (1 - rows["low_carbon_share"]).clip(lower=0)
    return rows


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in local time, because demand follows the Dutch clock, not UTC."""
    local = index.tz_convert(LOCAL_TZ)
    nl_holidays = holidays.country_holidays("NL", years=range(local.year.min(), local.year.max() + 1))
    is_holiday = np.array([day in nl_holidays for day in local.date])
    dow = local.dayofweek.to_numpy()
    # 0 = working day, 1 = Saturday, 2 = Sunday or public holiday
    day_type = np.where(is_holiday | (dow == 6), 2, np.where(dow == 5, 1, 0))
    doy = local.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "hour": local.hour.to_numpy(),
            "dow": dow,
            "day_type": day_type,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )


def local_midnights(first_day: str, last_day: str) -> pd.DatetimeIndex:
    """Forecast origins: every local midnight between two dates (inclusive), as UTC timestamps."""
    return pd.date_range(first_day, last_day, freq="D", tz=LOCAL_TZ).tz_convert("UTC")


def _padded(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.full(PAD, np.nan), np.asarray(values, dtype=float)])


def build_rows(
    grid: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizon: int = HORIZON,
    latency: int = LATENCY_HOURS,
) -> pd.DataFrame:
    """One row per (origin, hour ahead) with all features and the target y (NaN if unknown).

    `grid` must be a gap-free hourly UTC frame with columns ef, total and the LOW_CARBON series.
    For hours after the last actual, ef/total are NaN and the low-carbon columns hold forecasts.
    """
    step = np.diff(grid.index.to_numpy())  # resolution-agnostic (pandas 3 defaults to microseconds)
    if len(step) and (step != np.timedelta64(1, "h")).any():
        raise ValueError("grid must be a complete hourly index")
    grid = add_low_carbon(grid)
    positions = grid.index.get_indexer(origins)
    if (positions < 0).any():
        raise ValueError("every origin must be on the hourly grid")
    if (positions + horizon > len(grid)).any():
        raise ValueError("forecast window runs past the end of the grid")

    h = np.arange(horizon)
    target = positions[:, None] + h[None, :]  # (origins, horizon)
    cutoff = positions - latency  # first position hidden from the forecaster
    weeks_back = (h + latency) // WEEK + 1  # smallest whole number of weeks that lands before the cutoff
    t_pad = target + PAD

    ef = _padded(grid["ef"].to_numpy())
    total = _padded(grid["total"].to_numpy())

    def same_hour_weeks_ago(values: np.ndarray, extra: int) -> np.ndarray:
        return values[t_pad - WEEK * (weeks_back + extra)[None, :]]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        ef_profile = np.nanmean([same_hour_weeks_ago(ef, j) for j in range(PROFILE_WEEKS)], axis=0)
        total_profile = np.nanmean([same_hour_weeks_ago(total, j) for j in range(PROFILE_WEEKS)], axis=0)

    # Rolling statistics of the last visible day/week, read at the last visible hour (cutoff - 1)
    known = grid["ef"].notna() & grid["total"].notna() & grid["low_carbon"].notna()
    emissions = (grid["ef"] * grid["total"]).where(known)  # kg CO2 per hour
    residual = (grid["total"] - grid["low_carbon"]).where(known)  # MW not covered by low-carbon sources
    last = cutoff - 1 + PAD
    origin_stats = {
        "ef_last24": _padded(grid["ef"].rolling(24, min_periods=12).mean())[last],
        "ef_last168": _padded(grid["ef"].rolling(WEEK, min_periods=WEEK // 2).mean())[last],
        "intensity": _padded(
            emissions.rolling(WEEK, min_periods=WEEK // 2).sum()
            / residual.rolling(WEEK, min_periods=WEEK // 2).sum()
        )[last],
    }

    cal = calendar(grid.index)
    rows = {
        "origin": np.repeat(grid.index[positions], horizon),
        "time": grid.index[target.ravel()],
        "h": np.tile(h, len(positions)),
    }
    for column in CALENDAR:
        rows[column] = cal[column].to_numpy()[target].ravel()
    for column in LOW_CARBON:
        rows[column] = grid[column].to_numpy(dtype=float)[target].ravel()
    rows["ef_snaive"] = same_hour_weeks_ago(ef, 0).ravel()
    rows["ef_profile"] = ef_profile.ravel()
    for name, values in origin_stats.items():
        rows[name] = np.repeat(values, horizon)
    rows["total_profile"] = total_profile.ravel()

    out = add_derived(pd.DataFrame(rows))
    out["y"] = grid["ef"].to_numpy()[target].ravel()
    return out
