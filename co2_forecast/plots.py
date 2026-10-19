"""Report figures. One colour per entity across all figures; actuals are drawn in ink."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from co2_forecast.config import LOCAL_TZ

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {
    "lightgbm": "#2a78d6",
    "ridge": "#eb6834",
    "physics": "#1baf7a",
    "weekly_profile": "#eda100",
    "seasonal_naive": "#e87ba4",
    "ned": "#4a3aa7",
}
LABELS = {
    "lightgbm": "LightGBM (physics + correction)",
    "ridge": "Ridge",
    "physics": "Physics identity (no training)",
    "weekly_profile": "Weekly profile, last 4 weeks",
    "seasonal_naive": "Seasonal naive (same hour last week)",
    "ned": "NED's own forecast",
}
SERIES_ORDER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FEATURE_LABELS = {
    "ef_physics": "physics estimate of the CO$_2$ factor",
    "biomass": "biomass production",
    "low_carbon_share": "low-carbon share of typical production",
    "total_profile": "typical total production (4-week profile)",
    "intensity": "intensity of the rest of the mix, last week",
    "ef_profile": "CO$_2$ factor, same hour, 4-week mean",
    "doy_cos": "season (cosine of day of year)",
    "doy_sin": "season (sine of day of year)",
    "hour": "hour of day",
    "day_type": "working day / Saturday / Sunday-holiday",
    "dow": "day of week",
    "wind_onshore": "onshore wind production",
    "wind_offshore": "offshore wind production",
    "solar": "solar production",
    "low_carbon": "total low-carbon production",
    "ef_last168": "CO$_2$ factor, mean of last visible week",
    "ef_last24": "CO$_2$ factor, mean of last visible day",
    "ef_snaive": "CO$_2$ factor, same hour last week",
    "h": "hours ahead",
}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
EF_LABEL = "CO$_2$ factor, kg/kWh"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "text.color": INK,
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _local(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(LOCAL_TZ).tz_localize(None)


def eda_overview(hourly: pd.DataFrame, path: Path) -> Path:
    """CO2 factor and low-carbon share over time: daily means with a 30-day mean on top."""
    daily = hourly[["ef", "low_carbon", "total"]].resample("D").mean()
    daily.index = _local(daily.index)
    share = daily["low_carbon"] / daily["total"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    for ax, series, title, ylabel in (
        (axes[0], daily["ef"], "CO$_2$ factor of Dutch electricity production", "kg CO$_2$/kWh"),
        (axes[1], share, "Share of solar, wind and biomass in production", "share"),
    ):
        ax.plot(series.index, series, color=COLORS["lightgbm"], alpha=0.3, lw=0.8)
        rolling = series.rolling(30, center=True).mean()
        ax.plot(rolling.index, rolling, color=COLORS["lightgbm"], lw=1.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.annotate(
            "30-day mean",
            (rolling.index[-200], rolling.iloc[-200]),
            xytext=(0, 14),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            ha="center",
        )
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, path)


def eda_physics(hourly: pd.DataFrame, path: Path) -> Path:
    """Left: the accounting identity in the data. Right: daily shape of the CO2 factor by season."""
    recent = hourly.loc[hourly.index[-1] - pd.Timedelta(days=365) :].dropna(subset=["ef", "low_carbon"])
    share = recent["low_carbon"] / recent["total"]
    intensity = (recent["ef"] * recent["total"]).sum() / (recent["total"] - recent["low_carbon"]).sum()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.hexbin(share, recent["ef"], gridsize=45, mincnt=1, cmap=BLUES, linewidths=0.2, edgecolors=SURFACE)
    grid = np.linspace(0, 1, 50)
    ax.plot(grid, intensity * (1 - grid), color=INK, lw=1.2)
    ax.annotate(
        f"identity: {intensity:.2f} × (1 − share)",
        (0.52, intensity * 0.48),
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    ax.set_xlabel("low-carbon share of production (hourly)")
    ax.set_ylabel(EF_LABEL)
    ax.set_title("Last 12 months, hourly")
    ax.set_xlim(0, 1)

    ax = axes[1]
    local = recent.index.tz_convert(LOCAL_TZ)
    season = pd.Series(local.month, index=recent.index).map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )  # fmt: skip
    for color, name in zip(SERIES_ORDER, ["Winter", "Spring", "Summer", "Autumn"], strict=True):
        profile = recent["ef"][season == name].groupby(local.hour[season == name]).mean()
        ax.plot(profile.index, profile, color=color, label=name)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of day (local time)")
    ax.set_ylabel(EF_LABEL)
    ax.set_title("Average daily shape by season")
    ax.set_ylim(0, 0.4)
    ax.legend(ncols=4, loc="upper center", handlelength=1.2, columnspacing=1)
    fig.tight_layout()
    return _save(fig, path)


def lead_day_errors(scenarios: dict[str, pd.DataFrame], path: Path) -> Path:
    """MAE by lead day per model, one panel per scenario, shared y-axis."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(8, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, table) in zip(axes, scenarios.items(), strict=True):
        for model in table.columns:
            ax.plot(
                table.index,
                table[model],
                color=COLORS[model],
                marker="o",
                ms=4,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
                label=LABELS[model],
            )
        ax.set_title(title)
        ax.set_xlabel("lead day")
        ax.set_xticks(table.index)
    axes[0].set_ylim(0, max(table.to_numpy().max() for table in scenarios.values()) * 1.1)
    axes[0].set_ylabel("MAE, kg CO$_2$/kWh")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return _save(fig, path)


def example_weeks(preds: pd.DataFrame, model: str, bands: pd.DataFrame, path: Path) -> Path:
    """Best, typical and worst 7-day forecast of the test year (by MAE), with the 80% interval."""
    week_mae = preds.assign(err=(preds[model] - preds["y"]).abs()).groupby("origin")["err"].mean()
    week_mae = week_mae[preds.groupby("origin").size() == preds["h"].max() + 1]
    picks = {
        "Best week": week_mae.idxmin(),
        "Median week": week_mae.index[np.argsort(week_mae.to_numpy())[len(week_mae) // 2]],
        "Worst week": week_mae.idxmax(),
    }
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharey=True)
    for ax, (title, origin) in zip(axes, picks.items(), strict=True):
        week = preds[preds["origin"] == origin].sort_values("h")
        t = _local(pd.DatetimeIndex(week["time"]))
        lower = week[model] + bands["lower"].reindex(week["h"]).to_numpy()
        upper = week[model] + bands["upper"].reindex(week["h"]).to_numpy()
        ax.fill_between(t, lower, upper, color=COLORS[model], alpha=0.12, lw=0, label="80% interval")
        ax.plot(t, week["y"], color=INK, lw=1.2, label="actual")
        ax.plot(t, week[model], color=COLORS[model], label=LABELS[model])
        ax.set_title(f"{title}: from {origin.tz_convert(LOCAL_TZ):%a %d %b %Y}, MAE {week_mae[origin]:.3f}")
        ax.set_ylabel(EF_LABEL)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    axes[0].set_ylim(0, 0.55)
    axes[0].legend(ncols=3, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def feature_importance(single: pd.Series, groups: pd.Series, path: Path) -> Path:
    """Permutation importance: test MAE growth when a feature (left) or a group (right) is shuffled."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, importance, title, labels in (
        (axes[0], single.sort_values().tail(12), "Single features (top 12)", FEATURE_LABELS),
        (axes[1], groups.sort_values(), "Feature groups", {}),
    ):
        names = [labels.get(name, name) for name in importance.index]
        ax.barh(names, importance.to_numpy(), height=0.6, color=COLORS["lightgbm"])
        for y, value in enumerate(importance.to_numpy()):
            ax.annotate(
                f"{value:.3f}",
                (max(value, 0), y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=INK_2,
            )
        ax.set_xlim(0, importance.max() * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("MAE increase, kg CO$_2$/kWh")
    fig.tight_layout()
    return _save(fig, path)


def error_profile(by_hour: pd.DataFrame, by_month: pd.DataFrame, path: Path) -> Path:
    """Where the errors are: MAE by local hour of day and by calendar month."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, table, xlabel in ((axes[0], by_hour, "hour of day (local time)"), (axes[1], by_month, "month")):
        for model in table.columns:
            ax.plot(
                table.index,
                table[model],
                color=COLORS[model],
                marker="o",
                ms=3.5,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=LABELS[model],
            )
        ax.set_xlabel(xlabel)
    axes[0].set_ylim(0, max(by_hour.to_numpy().max(), by_month.to_numpy().max()) * 1.08)
    axes[0].set_xticks(range(0, 24, 3))
    axes[1].set_xticks(range(1, 13))
    axes[0].set_ylabel("MAE, kg CO$_2$/kWh")
    axes[0].set_title("By hour of day")
    axes[1].set_title("By month")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return _save(fig, path)


def forecast_plot(
    history: pd.DataFrame, forecast: pd.DataFrame, today: pd.DataFrame | None, path: Path
) -> Path:
    """The 7-day forecast with its 80% interval, NED's own forecast and the renewables it is based on."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    recent = history["ef"].iloc[-7 * 24 :]
    t_hist, t_fc = _local(recent.index), _local(forecast.index)
    ax.fill_between(
        t_fc,
        forecast["lower"],
        forecast["upper"],
        color=COLORS["lightgbm"],
        alpha=0.12,
        lw=0,
        label="80% interval",
    )
    ax.plot(t_hist, recent, color=INK, lw=1.2, label="actual")
    if today is not None and len(today):
        ax.plot(_local(today.index), today["ef"], color=INK, lw=1.2)
    ax.plot(t_fc, forecast["ned_forecast"], color=COLORS["ned"], lw=1.2, label=LABELS["ned"])
    ax.plot(t_fc, forecast["forecast"], color=COLORS["lightgbm"], label=LABELS["lightgbm"])
    gap_middle = t_hist[-1] + (t_fc[0] - t_hist[-1]) / 2
    ax.annotate("publication\ndelay", (gap_middle, 0.03), ha="center", fontsize=7, color=MUTED)
    ax.set_ylabel(EF_LABEL)
    ax.set_ylim(0, 0.6)
    ax.set_title(
        f"CO$_2$ factor forecast from {forecast.index[0].tz_convert(LOCAL_TZ):%a %d %b %Y}, local time"
    )
    ax.legend(ncols=4, loc="upper left")

    ax = axes[1]
    inputs = {
        "solar": (forecast["solar"], "#eda100"),
        "wind (onshore + offshore)": (forecast["wind_onshore"] + forecast["wind_offshore"], "#1baf7a"),
        "biomass": (forecast["biomass"], "#e87ba4"),
    }
    for label, (series, color) in inputs.items():
        ax.plot(t_fc, series / 1000, color=color, label=label)
    ax.set_ylabel("GW")
    ax.set_ylim(0, max(series.max() for series, _ in inputs.values()) / 1000 * 1.3)
    ax.set_title("NED forecast of low-carbon production (model input)")
    ax.legend(ncols=3, loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    fig.tight_layout()
    return _save(fig, path)
