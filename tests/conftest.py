import numpy as np
import pandas as pd
import pytest


def synthetic_grid(days: int = 84, seed: int = 0) -> pd.DataFrame:
    """Hourly grid that behaves like NED data: CO2 factor = intensity * (1 - low-carbon share) + noise."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * 24, freq="h", tz="UTC")
    t = np.arange(len(index))
    hour = index.tz_convert("Europe/Amsterdam").hour.to_numpy()
    solar = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None) * 6000
    wind_onshore = 2000 + 1500 * np.sin(t / 41.0) + rng.normal(0, 200, len(t)).clip(-1500)
    wind_offshore = 1800 + 1200 * np.sin(t / 53.0 + 1)
    biomass = 450 + 100 * np.sin(t / 97.0)
    total = 13000 + 2000 * np.sin((hour - 4) / 24 * 2 * np.pi)
    low_carbon = solar + wind_onshore + wind_offshore + biomass
    ef = 0.42 * np.clip(1 - low_carbon / total, 0, None) + rng.normal(0, 0.005, len(t))
    return pd.DataFrame(
        {
            "total": total,
            "ef": ef,
            "solar": solar,
            "wind_onshore": wind_onshore,
            "wind_offshore": wind_offshore,
            "biomass": biomass,
        },
        index=index,
    )


@pytest.fixture
def grid() -> pd.DataFrame:
    return synthetic_grid()
