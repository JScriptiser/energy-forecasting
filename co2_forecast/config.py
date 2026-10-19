"""Project-wide constants. Everything that defines the forecasting problem lives here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "ned"
OUTPUT_DIR = ROOT / "output"

LOCAL_TZ = "Europe/Amsterdam"

# 7 days ahead at hourly resolution = 168 values per forecast
HORIZON = 168

# NED publishes the hourly yearly files with a delay: when the forecast for day D is made
# (origin = local midnight of D), actuals are only available up to the end of day D-2.
# The backtest reproduces the same information set, so the model never sees the last 24 hours.
LATENCY_HOURS = 24

# Mapping of our column names to NED file prefixes (yearly/today files, 7-day forecast files)
# and the NED series type expected inside each file.
SERIES = {
    "mix": {
        "history": ("electriciteitsmix",),
        "forecast": ("totaal-elektriciteitsmix",),
        "type": "ElectricityMix",
    },
    "solar": {"history": ("zon",), "forecast": ("zon",), "type": "Solar"},
    "wind_onshore": {"history": ("wind",), "forecast": ("wind",), "type": "Wind"},
    "wind_offshore": {"history": ("zeewind",), "forecast": ("zeewind",), "type": "WindOffshoreC"},
    "biomass": {"history": ("biomassa",), "forecast": ("biomassa",), "type": "BiomassPower"},
}

# The CO2 factor series starts in 2021, so older renewables files are not needed.
FIRST_YEAR = 2021

RENEWABLES = ("solar", "wind_onshore", "wind_offshore")
LOW_CARBON = (*RENEWABLES, "biomass")

# Backtest layout: hyperparameters and features are chosen on the validation year,
# the test year is touched once for the final comparison.
VALIDATION_START = "2024-10-15"
TEST_START = "2025-10-15"
REFIT_EVERY_DAYS = 28

SEED = 42
