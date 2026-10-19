"""Reading and validating NED (Nationaal Energie Dashboard) CSV exports.

NED exports one CSV per series and year ("zon-2025-uur-data.csv"), a file with today's hours
("zon-afgelopen-dag.csv") and a 7-day forecast ("zon-uur-voorspelling-aankomende-7-dagen.csv").
All of them share the same 13-column layout with timestamps in UTC.

The loaders return hourly frames indexed by UTC timestamps with average power in MW
(NED reports energy per hour in kWh, so kWh / 1000 = MW) and the emission factor in kg CO2/kWh.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from co2_forecast.config import FIRST_YEAR, RAW_DIR, SERIES

COLUMNS = {
    "validfrom (UTC)": "validfrom",
    "validto (UTC)": "validto",
    "point": "point",
    "type": "type",
    "granularity": "granularity",
    "timezone": "timezone",
    "activity": "activity",
    "classification": "classification",
    "volume (kWh)": "volume_kwh",
    "emission (kg CO2)": "emission_kg",
    "emissionfactor (kg CO2/kWh)": "emissionfactor",
}

# Fixed metadata every hourly NED file must have
EXPECTED = {"point": "Nederland", "granularity": "Hour", "timezone": "UTC", "activity": "Providing"}

# Plausibility limits; values outside are treated as data errors
MAX_EMISSION_FACTOR = 1.5  # kg CO2/kWh, above pure lignite
MAX_POWER_MW = 60_000  # above the total installed capacity of the Netherlands


@dataclass
class DataReport:
    """What the loader found and fixed, written to output/data_issues.csv."""

    issues: list[dict] = field(default_factory=list)

    def add(self, source: str, issue: str, count: int, action: str) -> None:
        if count:
            self.issues.append({"source": source, "issue": issue, "count": int(count), "action": action})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=["source", "issue", "count", "action"])


def _files(raw_dir: Path, prefixes: tuple[str, ...], kind: str) -> list[Path]:
    """Find the files of one series. Browser duplicates like 'x (1).csv' are skipped on purpose."""
    alternatives = "|".join(re.escape(p) for p in prefixes)
    patterns = {
        "history": rf"^(?:{alternatives})-(\d{{4}})-uur-data\.(?:csv|txt)$",
        "today": rf"^(?:{alternatives})-afgelopen-dag\.csv$",
        "forecast": rf"^(?:{alternatives})-uur-voorspelling-aankomende-7-dagen\.csv$",
    }
    regex = re.compile(patterns[kind], re.IGNORECASE)
    found = []
    for path in sorted(raw_dir.iterdir()):
        match = regex.match(path.name)
        if match and (kind != "history" or int(match.group(1)) >= FIRST_YEAR):
            found.append(path)
    return found


def read_ned_csv(path: Path, expected_type: str, classification: str) -> pd.DataFrame:
    """Read one NED file and check that it contains what its name promises."""
    df = pd.read_csv(path, usecols=lambda c: c in COLUMNS).rename(columns=COLUMNS)
    missing = {"validfrom", "validto", "type", "volume_kwh"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    checks = {**EXPECTED, "type": expected_type, "classification": classification}
    for column, value in checks.items():
        bad = df[column].astype(str) != value
        if bad.any():
            raise ValueError(f"{path.name}: {bad.sum()} rows with {column} != {value!r}")

    df["validfrom"] = pd.to_datetime(df["validfrom"], utc=True)
    df["validto"] = pd.to_datetime(df["validto"], utc=True)
    if ((df["validto"] - df["validfrom"]) != pd.Timedelta(hours=1)).any():
        raise ValueError(f"{path.name}: rows that do not span exactly one hour")
    return df


def _to_hourly(frames: list[pd.DataFrame], name: str, report: DataReport) -> pd.DataFrame:
    """Concatenate files of one series, drop duplicates and implausible values."""
    df = pd.concat(frames, ignore_index=True).sort_values("validfrom")
    duplicated = df.duplicated("validfrom", keep="last")
    report.add(name, "duplicate timestamps", duplicated.sum(), "kept the latest file's value")
    df = df[~duplicated].set_index("validfrom")

    out = pd.DataFrame(index=df.index)
    out[name] = df["volume_kwh"].astype(float) / 1000.0  # kWh per hour -> average MW
    bad_power = (out[name] < 0) | (out[name] > MAX_POWER_MW)
    report.add(name, "power outside [0, 60 GW]", bad_power.sum(), "set to NaN")
    out.loc[bad_power, name] = np.nan

    if name == "mix":
        out = out.rename(columns={"mix": "total"})
        ef = df["emissionfactor"].astype(float)
        # NED writes 0 when no emission factor was computed; a real mix is never CO2-free
        bad_ef = (ef <= 0) | (ef > MAX_EMISSION_FACTOR) | ef.isna()
        report.add(name, "emission factor missing, zero or > 1.5", bad_ef.sum(), "set to NaN")
        out["ef"] = ef.mask(bad_ef)
    return out


def load_history(raw_dir: Path = RAW_DIR, report: DataReport | None = None) -> pd.DataFrame:
    """Hourly actuals of all series on a complete UTC grid (missing hours are NaN).

    Columns: ef (kg CO2/kWh), total, solar, wind_onshore, wind_offshore, biomass (MW).
    """
    report = report if report is not None else DataReport()
    parts = []
    for name, spec in SERIES.items():
        paths = _files(raw_dir, spec["history"], "history")
        if not paths:
            raise FileNotFoundError(f"no yearly NED files for {name!r} in {raw_dir}")
        frames = [read_ned_csv(p, spec["type"], "Current") for p in paths]
        parts.append(_to_hourly(frames, name, report))

    hourly = pd.concat(parts, axis=1).sort_index()
    # The CO2 factor is the target: the usable period is where it exists
    hourly = hourly.loc[hourly["ef"].first_valid_index() : hourly["ef"].last_valid_index()]
    grid = pd.date_range(hourly.index[0], hourly.index[-1], freq="h", tz="UTC")
    report.add("all", "hours missing from the files", len(grid) - len(hourly), "left as NaN")
    hourly = hourly.reindex(grid)
    hourly.index.name = "time"

    for column in hourly.columns:
        report.add(column, "missing values", hourly[column].isna().sum(), "handled by the feature builder")
    return hourly


def load_today(raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    """Actuals of the current day ('afgelopen-dag' files). Used only to check the live forecast."""
    parts = []
    for name, spec in SERIES.items():
        paths = _files(raw_dir, spec["history"], "today")
        if paths:
            parts.append(_to_hourly([read_ned_csv(paths[0], spec["type"], "Current")], name, DataReport()))
    if not parts:
        return None
    today = pd.concat(parts, axis=1).sort_index()
    today.index.name = "time"
    return today


def load_ned_forecast(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """NED's own 7-day hourly forecasts: renewables (our model inputs) and the mix (a benchmark).

    Columns: solar, wind_onshore, wind_offshore, biomass (MW), total (MW) and ef (NED's CO2 factor forecast).
    """
    parts = []
    for name, spec in SERIES.items():
        paths = _files(raw_dir, spec["forecast"], "forecast")
        if not paths:
            raise FileNotFoundError(f"no 7-day forecast file for {name!r} in {raw_dir}")
        parts.append(_to_hourly([read_ned_csv(paths[0], spec["type"], "Forecast")], name, DataReport()))
    forecast = pd.concat(parts, axis=1).sort_index()
    if forecast.isna().any().any():
        raise ValueError("NED forecast files do not cover the same hours")
    forecast.index.name = "time"
    return forecast
