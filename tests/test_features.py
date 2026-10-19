import numpy as np
import pandas as pd
import pytest

from co2_forecast.config import HORIZON, LATENCY_HOURS
from co2_forecast.features import ALL_FEATURES, DERIVED, HISTORY, build_rows, calendar, local_midnights


def test_one_row_per_origin_and_hour(grid):
    origins = local_midnights("2025-02-15", "2025-02-20")
    rows = build_rows(grid, origins)
    assert len(rows) == len(origins) * HORIZON
    assert set(ALL_FEATURES) <= set(rows.columns)
    assert (rows.groupby("origin")["h"].apply(list) == [list(range(HORIZON))] * len(origins)).all()


def test_nothing_at_or_after_the_cutoff_reaches_the_history_features(grid):
    """Only hours before origin - LATENCY_HOURS are visible; scrambling later values must not matter."""
    origin = local_midnights("2025-03-01", "2025-03-01")
    before = build_rows(grid, origin)

    tampered = grid.copy()
    hidden = tampered.index >= origin[0] - pd.Timedelta(hours=LATENCY_HOURS)
    tampered.loc[hidden, ["ef", "total"]] = 99.0
    after = build_rows(tampered, origin)

    pd.testing.assert_frame_equal(before[HISTORY + DERIVED], after[HISTORY + DERIVED])
    assert (after["y"] == 99.0).all()  # the target itself did change


def test_history_features_change_when_visible_data_changes(grid):
    origin = local_midnights("2025-03-01", "2025-03-01")
    before = build_rows(grid, origin)
    tampered = grid.copy()
    visible = tampered.index < origin[0] - pd.Timedelta(hours=LATENCY_HOURS)
    tampered.loc[visible, "ef"] += 0.1
    after = build_rows(tampered, origin)
    assert np.allclose(after["ef_last168"], before["ef_last168"] + 0.1)


def test_seasonal_naive_value_is_the_latest_visible_same_hour(grid):
    origin = local_midnights("2025-03-01", "2025-03-01")
    rows = build_rows(grid, origin)
    cutoff = origin[0] - pd.Timedelta(hours=LATENCY_HOURS)
    for h in (0, 100, 143, 144, 167):
        target = rows.loc[rows["h"] == h, "time"].iloc[0]
        weeks = 1 if target - pd.Timedelta(days=7) < cutoff else 2
        expected = grid.loc[target - pd.Timedelta(days=7 * weeks), "ef"]
        assert rows.loc[rows["h"] == h, "ef_snaive"].iloc[0] == pytest.approx(expected)


def test_calendar_is_in_dutch_local_time():
    index = pd.DatetimeIndex(["2026-07-01 10:00", "2026-04-27 10:00", "2026-04-28 10:00"], tz="UTC")
    cal = calendar(index)
    assert cal["hour"].tolist() == [12, 12, 12]  # CEST = UTC+2
    assert cal["day_type"].tolist() == [0, 2, 0]  # King's Day is a public holiday


def test_grid_with_gaps_is_rejected(grid):
    with pytest.raises(ValueError, match="complete hourly"):
        build_rows(grid.drop(grid.index[500]), local_midnights("2025-02-15", "2025-02-15"))


def test_window_past_the_end_is_rejected(grid):
    last_day = str(grid.index[-1].date())
    with pytest.raises(ValueError, match="past the end"):
        build_rows(grid, local_midnights(last_day, last_day))
