import pandas as pd
import pytest

from co2_forecast.data import DataReport, load_history, load_ned_forecast

HEADER = (
    '"validfrom (UTC)","validto (UTC)",point,type,granularity,timezone,activity,classification,'
    '"capacity (kW)","volume (kWh)",percentage,"emission (kg CO2)","emissionfactor (kg CO2/kWh)"'
)
TYPES = {
    "electriciteitsmix": "ElectricityMix",
    "zon": "Solar",
    "wind": "Wind",
    "zeewind": "WindOffshoreC",
    "biomassa": "BiomassPower",
}


def write_ned(path, ned_type, start, hours, classification="Current", volume=1_000_000, ef=0.3):
    lines = [HEADER]
    for t in pd.date_range(start, periods=hours, freq="h"):
        end = t + pd.Timedelta(hours=1)
        lines.append(
            f'"{t:%Y-%m-%d %H:%M:%S}","{end:%Y-%m-%d %H:%M:%S}",Nederland,{ned_type},Hour,UTC,Providing,'
            f"{classification},{volume},{volume},0.1,{volume * ef},{ef}"
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def raw_dir(tmp_path):
    for prefix, ned_type in TYPES.items():
        write_ned(tmp_path / f"{prefix}-2025-uur-data.csv", ned_type, "2024-12-31 23:00", 48)
    return tmp_path


def test_history_is_merged_on_an_hourly_utc_grid_in_mw(raw_dir):
    hourly = load_history(raw_dir)
    assert list(hourly.columns) == ["total", "ef", "solar", "wind_onshore", "wind_offshore", "biomass"]
    assert len(hourly) == 48
    assert str(hourly.index.tz) == "UTC"
    assert hourly["total"].iloc[0] == pytest.approx(1000.0)  # 1,000,000 kWh in one hour = 1000 MW
    assert hourly["ef"].iloc[0] == pytest.approx(0.3)


def test_browser_duplicates_are_ignored(raw_dir):
    write_ned(raw_dir / "zon-2025-uur-data (1).csv", "Solar", "2024-12-31 23:00", 48, volume=5)
    assert load_history(raw_dir)["solar"].iloc[0] == pytest.approx(1000.0)


def test_zero_emission_factor_becomes_missing_and_is_reported(raw_dir):
    write_ned(raw_dir / "electriciteitsmix-2025-uur-data.csv", "ElectricityMix", "2024-12-31 23:00", 48, ef=0)
    write_ned(raw_dir / "electriciteitsmix-2024-uur-data.csv", "ElectricityMix", "2024-12-31 21:00", 2)
    report = DataReport()
    hourly = load_history(raw_dir, report)
    # the all-zero tail is trimmed: the usable period ends at the last real value
    assert hourly["ef"].isna().sum() == 0
    issues = report.to_frame()
    assert (issues["issue"] == "emission factor missing, zero or > 1.5").any()


def test_a_file_with_the_wrong_series_is_rejected(raw_dir):
    write_ned(raw_dir / "zon-2025-uur-data.csv", "Wind", "2024-12-31 23:00", 48)
    with pytest.raises(ValueError, match="type"):
        load_history(raw_dir)


def test_forecast_files_must_cover_the_same_hours(raw_dir):
    for prefix, ned_type in TYPES.items():
        name = "totaal-elektriciteitsmix" if prefix == "electriciteitsmix" else prefix
        write_ned(
            raw_dir / f"{name}-uur-voorspelling-aankomende-7-dagen.csv",
            ned_type,
            "2025-01-05 23:00",
            168,
            classification="Forecast",
        )
    forecast = load_ned_forecast(raw_dir)
    assert len(forecast) == 168 and not forecast.isna().any().any()

    write_ned(
        raw_dir / "zon-uur-voorspelling-aankomende-7-dagen.csv",
        "Solar",
        "2025-01-06 23:00",
        168,
        classification="Forecast",
    )
    with pytest.raises(ValueError, match="same hours"):
        load_ned_forecast(raw_dir)
