"""Everything together: data checks, EDA figures, backtests, the final forecast. Results go to output/.

python -m co2_forecast             full run: backtests + forecast (about 3 minutes)
python -m co2_forecast forecast    only refit and forecast the next 168 hours
python -m co2_forecast evaluate    score a saved forecast against actuals downloaded later
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from co2_forecast import plots
from co2_forecast.config import (
    LATENCY_HOURS,
    LOCAL_TZ,
    LOW_CARBON,
    OUTPUT_DIR,
    RAW_DIR,
    SEED,
    TEST_START,
    VALIDATION_START,
)
from co2_forecast.data import DataReport, load_history, load_ned_forecast, load_today
from co2_forecast.evaluation import (
    backtest,
    interval_coverage,
    lead_day,
    mae_by,
    metrics,
    residual_quantiles,
    summarize,
    with_forecast_errors,
)
from co2_forecast.features import ALL_FEATURES, add_derived, add_low_carbon, build_rows, local_midnights
from co2_forecast.models import BASELINE, MODELS, LightGBMModel

FINAL_MODEL = "lightgbm"
TRAIN_START = "2021-02-15"  # first origin with five weeks of history behind it
TRAIN_STRIDE_DAYS = 2  # neighbouring origins share most target hours; every second one is enough
SCENARIOS = {"perfect_renewables": 0.0, "forecast_errors": 1.0, "forecast_errors_x2": 2.0}
STARTED = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - STARTED:6.1f}s] {message}", flush=True)


def training_rows(grid: pd.DataFrame, last_origin: pd.Timestamp) -> pd.DataFrame:
    """All (origin, hour) rows with a known target, origins every TRAIN_STRIDE_DAYS up to last_origin."""
    origins = local_midnights(TRAIN_START, str(last_origin.tz_convert(LOCAL_TZ).date()))[::TRAIN_STRIDE_DAYS]
    rows = build_rows(grid, origins)
    return rows[rows["y"].notna()]


# Information sources for the grouped importance; derived features are recomputed after shuffling
FEATURE_GROUPS = {
    "low-carbon production\n(NED forecasts)": list(LOW_CARBON),
    "history visible\nat the origin": [
        "ef_snaive",
        "ef_profile",
        "ef_last24",
        "ef_last168",
        "total_profile",
        "intensity",
    ],
    "calendar": ["hour", "dow", "day_type", "doy_sin", "doy_cos"],
    "hours ahead": ["h"],
}


def permutation_importance(
    model, rows: pd.DataFrame, groups: dict[str, list[str]], n_repeats: int = 3
) -> pd.Series:
    """Mean increase of MAE when the columns of a group are shuffled together (the model is not refitted)."""
    rng = np.random.default_rng(SEED)
    y = rows["y"].to_numpy()
    base = metrics(y, model.predict(rows))["MAE"]
    importance = {}
    for name, columns in groups.items():
        increases = []
        for _ in range(n_repeats):
            shuffled = rows.copy()
            shuffled[columns] = rows[columns].to_numpy()[rng.permutation(len(rows))]
            if set(columns) & {*LOW_CARBON, "total_profile", "intensity"} and len(columns) > 1:
                add_derived(shuffled)
            increases.append(metrics(y, model.predict(shuffled))["MAE"] - base)
        importance[name] = float(np.mean(increases))
    return pd.Series(importance, name="mae_increase")


def run_eda(hourly: pd.DataFrame, out: Path) -> None:
    hourly = add_low_carbon(hourly)
    year = hourly.index.tz_convert(LOCAL_TZ).year
    summary = hourly.groupby(year).agg(
        hours=("ef", "size"),
        ef_mean=("ef", "mean"),
        ef_p10=("ef", lambda s: s.quantile(0.1)),
        ef_p90=("ef", lambda s: s.quantile(0.9)),
        total_gw=("total", lambda s: s.mean() / 1000),
        low_carbon_gw=("low_carbon", lambda s: s.mean() / 1000),
    )
    summary["low_carbon_share"] = summary["low_carbon_gw"] / summary["total_gw"]
    summary.round(4).to_csv(out / "data_summary.csv", index_label="year")
    share = hourly["low_carbon"] / hourly["total"]
    log(f"correlation of the CO2 factor with the low-carbon share: {hourly['ef'].corr(share):.3f}")
    plots.eda_overview(hourly, out / "figures" / "eda_overview.png")
    plots.eda_physics(hourly, out / "figures" / "eda_physics.png")


def run_backtests(grid: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Validation year (choices were made there) and test year, under each renewables scenario."""
    validation_end = (pd.Timestamp(TEST_START, tz=LOCAL_TZ) - pd.Timedelta(days=7)).tz_convert("UTC")
    periods = {"validation": (VALIDATION_START, validation_end), "test": (TEST_START, None)}
    names = list(MODELS)
    summary, by_day, results = [], [], {}
    for period, (start, end) in periods.items():
        for scenario, noise in SCENARIOS.items():
            if period == "validation" and scenario == "forecast_errors_x2":
                continue
            log(f"backtest: {period} year, {scenario}")
            preds = backtest(
                grid,
                MODELS,
                TRAIN_START,
                start,
                test_end=end,
                train_stride_days=TRAIN_STRIDE_DAYS,
                test_noise=noise,
                log=lambda _: None,
            )
            results[period, scenario] = preds
            table = summarize(preds, names, BASELINE)
            summary.append(table.assign(period=period, scenario=scenario).rename_axis("model").reset_index())
            days = mae_by(preds, names, lead_day(preds)).rename_axis("lead_day").reset_index()
            by_day.append(
                days.melt("lead_day", var_name="model", value_name="MAE").assign(
                    period=period, scenario=scenario
                )
            )
            mae = table["MAE"]
            log(f"  {FINAL_MODEL} MAE {mae[FINAL_MODEL]:.4f}, {BASELINE} {mae[BASELINE]:.4f}")

    summary = pd.concat(summary)[
        ["period", "scenario", "model", "MAE", "RMSE", "WAPE_%", "bias", f"skill_vs_{BASELINE}_%"]
    ]
    summary.round(5).to_csv(out / "backtest_summary.csv", index=False)
    pd.concat(by_day)[["period", "scenario", "model", "lead_day", "MAE"]].round(5).to_csv(
        out / "backtest_by_lead_day.csv", index=False
    )

    # 80% interval: residual quantiles per hour ahead, calibrated on validation, checked on test.
    # The forecast uses the quantiles of the most recent year, rounded as stored, so that
    # `python -m co2_forecast forecast` reproduces the full run exactly.
    calibration = residual_quantiles(results["validation", "forecast_errors"], FINAL_MODEL)
    coverage = interval_coverage(results["test", "forecast_errors"], FINAL_MODEL, calibration)
    log(f"80% interval calibrated on validation covers {coverage:.1%} of the test year")
    bands = residual_quantiles(results["test", "forecast_errors"], FINAL_MODEL).round(5)
    bands.to_csv(out / "interval_quantiles.csv")
    pd.DataFrame({"interval": ["80%"], "test_coverage": [round(coverage, 4)]}).to_csv(
        out / "interval_coverage.csv", index=False
    )

    test = results["test", "forecast_errors"]
    test[["origin", "time", "h", "y", *names]].round(dict.fromkeys(["y", *names], 5)).to_csv(
        out / "backtest_test_predictions.csv.gz", index=False
    )
    local = pd.DatetimeIndex(test["time"]).tz_convert(LOCAL_TZ)
    shown = [FINAL_MODEL, "physics", BASELINE]
    by_hour = mae_by(test, shown, pd.Series(local.hour))
    by_month = mae_by(test, shown, pd.Series(local.month))
    plots.error_profile(by_hour, by_month, out / "figures" / "error_profile.png")
    plots.example_weeks(test, FINAL_MODEL, bands, out / "figures" / "example_weeks.png")
    plots.lead_day_errors(
        {
            "Actual renewables as inputs (upper bound)": mae_by(
                results["test", "perfect_renewables"], names, lead_day(results["test", "perfect_renewables"])
            ),
            "Synthetic forecast errors (realistic)": mae_by(test, names, lead_day(test)),
        },
        out / "figures" / "lead_day_errors.png",
    )

    # Feature importance on the test year for the model trained on everything before it
    test_start = pd.Timestamp(TEST_START, tz=LOCAL_TZ)
    train = training_rows(grid, test_start - pd.Timedelta(days=1))
    train = train[train["time"] < test_start - pd.Timedelta(hours=LATENCY_HOURS)]
    model = LightGBMModel().fit(train)
    test_rows = test[["origin", "h"]].merge(
        with_forecast_errors(build_rows(grid, pd.DatetimeIndex(test["origin"].unique())), 1.0, SEED + 1),
        on=["origin", "h"],
    )
    single = permutation_importance(model, test_rows, {feature: [feature] for feature in ALL_FEATURES})
    groups = permutation_importance(model, test_rows, FEATURE_GROUPS)
    single.sort_values(ascending=False).round(5).to_csv(out / "feature_importance.csv", index_label="feature")
    groups.rename(lambda name: name.replace("\n", " ")).sort_values(ascending=False).round(5).to_csv(
        out / "feature_group_importance.csv", index_label="group"
    )
    plots.feature_importance(single, groups, out / "figures" / "feature_importance.png")
    return bands


def run_forecast(grid: pd.DataFrame, out: Path, bands: pd.DataFrame | None) -> pd.DataFrame:
    """Refit on all history and forecast the 168 hours covered by NED's latest renewables forecast."""
    ned = load_ned_forecast()
    origin = ned.index[0]
    last_actual = grid["ef"].last_valid_index()
    gap = int((origin - last_actual) / pd.Timedelta(hours=1)) - 1
    log(f"forecast origin {origin.tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M %Z}, last actual {gap}h before it")
    if gap > LATENCY_HOURS:
        log(f"WARNING: history ends {gap}h before the origin, the model was trained for {LATENCY_HOURS}h")

    full_index = pd.date_range(grid.index[0], ned.index[-1], freq="h", tz="UTC")
    full = grid.reindex(full_index)
    future = full.index >= origin
    full.loc[future, list(LOW_CARBON)] = ned[list(LOW_CARBON)].to_numpy()
    full.loc[future, ["ef", "total"]] = np.nan  # never let actuals inside the window reach the model

    model = LightGBMModel().fit(training_rows(full, origin - pd.Timedelta(days=1)))
    rows = build_rows(full, pd.DatetimeIndex([origin]))
    prediction = model.predict(rows)
    forecast = pd.DataFrame(index=ned.index)
    forecast.index.name = "time_utc"
    forecast["time_local"] = ned.index.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    forecast["lead_hour"] = rows["h"].to_numpy() + 1
    forecast["forecast"] = prediction
    if bands is not None:
        forecast["lower"] = np.maximum(prediction + bands["lower"].reindex(rows["h"]).to_numpy(), 0)
        forecast["upper"] = prediction + bands["upper"].reindex(rows["h"]).to_numpy()
    else:
        forecast["lower"] = forecast["upper"] = np.nan
    forecast["ned_forecast"] = ned["ef"]
    for column in LOW_CARBON:
        forecast[column] = ned[column]
    forecast.round(5).to_csv(out / "forecast.csv")
    means = forecast[["forecast", "ned_forecast"]].mean()
    log(f"forecast: mean {means['forecast']:.3f} kg/kWh (NED's own: {means['ned_forecast']:.3f})")

    today = load_today()
    if today is not None:
        overlap = forecast.join(today[["ef"]].rename(columns={"ef": "actual"}), how="inner").dropna(
            subset=["actual"]
        )
        if len(overlap):
            check = pd.DataFrame(
                {
                    "hours": [len(overlap)] * 2,
                    "model": [FINAL_MODEL, "ned_forecast"],
                    "MAE": [
                        metrics(overlap["actual"].to_numpy(), overlap["forecast"].to_numpy())["MAE"],
                        metrics(overlap["actual"].to_numpy(), overlap["ned_forecast"].to_numpy())["MAE"],
                    ],
                }
            )
            check.round(5).to_csv(out / "live_check.csv", index=False)
            log(
                f"first {len(overlap)} forecast hours already happened: MAE ours {check['MAE'][0]:.4f}, "
                f"NED {check['MAE'][1]:.4f} (too few hours to compare models)"
            )
    plots.forecast_plot(grid, forecast, today, out / "figures" / "forecast.png")
    return forecast


def evaluate_saved(out: Path) -> None:
    """Score output/forecast.csv against actuals from a newer NED download."""
    forecast = pd.read_csv(out / "forecast.csv", index_col="time_utc")
    forecast.index = pd.to_datetime(forecast.index, utc=True)
    actual = load_history()["ef"]
    today = load_today()
    if today is not None:
        actual = actual.combine_first(today["ef"])
    joined = forecast.join(actual.rename("actual"), how="inner").dropna(subset=["actual"])
    if joined.empty:
        log("no actuals for the forecast window yet: download newer NED files into data/raw/ned")
        return
    rows = []
    for model, column in ((FINAL_MODEL, "forecast"), ("ned_forecast", "ned_forecast")):
        rows.append(
            {
                "model": model,
                "hours": len(joined),
                **metrics(joined["actual"].to_numpy(), joined[column].to_numpy()),
            }
        )
    result = pd.DataFrame(rows)
    result.round(5).to_csv(out / "live_evaluation.csv", index=False)
    log("\n" + result.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m co2_forecast",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="all", choices=["all", "forecast", "evaluate"])
    args = parser.parse_args()
    out = OUTPUT_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)

    if args.command == "evaluate":
        evaluate_saved(out)
        return

    report = DataReport()
    hourly = load_history(RAW_DIR, report)
    report.to_frame().to_csv(out / "data_issues.csv", index=False)
    log(
        f"history: {len(hourly):,} hours, {hourly.index[0]:%Y-%m-%d} .. {hourly.index[-1]:%Y-%m-%d %H:%M} UTC"
    )

    bands = None
    if args.command == "all":
        run_eda(hourly, out)
        bands = run_backtests(hourly, out)
    elif (out / "interval_quantiles.csv").exists():
        bands = pd.read_csv(out / "interval_quantiles.csv", index_col="h")
    run_forecast(hourly, out, bands)
    log(f"done, results in {out.relative_to(out.parent)}/")


if __name__ == "__main__":
    main()
