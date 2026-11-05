"""Charts for the washing machine analysis. One hue per measure, with houses as small multiples."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.analysis.usage import WEEKDAYS  # noqa: E402

# Colours passed the dataviz palette validator on the light surface.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e5e2", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NEUTRAL = "#8c8b86"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(rows: int, cols: int, width: float = 13, height: float = 2.6, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height * rows), constrained_layout=True, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        style(ax)
    return fig, np.atleast_1d(axes)


def finish(fig, title: str, path: Path) -> Path:
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def representative_traces(days: dict[int, pd.Timestamp], tables: dict[int, pd.DataFrame], path: Path) -> Path:
    fig, axes = figure(3, 2, height=2.5, sharey=False)
    for ax, (house, day) in zip(axes.ravel(), days.items()):
        table = tables[house]
        rows = table[(table["uk_local_minute"] >= day) & (table["uk_local_minute"] < day + pd.Timedelta(days=1))]
        ax.plot(rows["uk_local_minute"], rows["aggregate_w"], color=BLUE, lw=0.9, label="Household aggregate")
        ax.plot(rows["uk_local_minute"], rows["washing_machine_w"], color=ORANGE, lw=1.1, label="Washing machine")
        ax.set_title(f"House {house}, {day:%a %d %b %Y}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(day, day + pd.Timedelta(days=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
        ax.set_ylabel("Watts", fontsize=8, color=MUTED)
        ax.set_ylim(bottom=0)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=2)
    return finish(fig, "One typical day per household, UK local time", path)


def cycle_examples(examples: pd.DataFrame, tables: dict[int, pd.DataFrame], path: Path) -> Path:
    houses = list(tables)
    fig, axes = figure(len(houses), 4, height=1.75, sharey=True)
    for row, house in zip(axes, houses):
        chosen = examples[examples["house"] == house]
        table = tables[house]
        for ax, (_, cycle) in zip(row, chosen.iterrows()):
            start, end = int(cycle["start_pos"]) - 5, int(cycle["end_pos"]) + 5
            power = table["washing_machine_w"].to_numpy()[max(start, 0):end + 1]
            ax.plot(np.arange(len(power)) - 5, power, color=ORANGE, lw=1.0)
            ax.set_title(f"House {house}: {cycle['duration_min']:.0f} min, {cycle['energy_kwh']:.2f} kWh, peak {cycle['peak_w']:,.0f} W",
                         fontsize=7.8, color=INK, loc="left")
            ax.set_ylim(-40, 3000)
    for ax in axes[:, 0]:
        ax.set_ylabel("Watts", fontsize=8, color=MUTED)
    for ax in axes[-1, :]:
        ax.set_xlabel("Minutes from cycle start", fontsize=8, color=MUTED)
    return finish(fig, "Detected washing machine cycles, spread across each household's duration range", path)


def hourly_distribution(hourly: pd.DataFrame, path: Path) -> Path:
    houses = sorted(hourly["house"].unique())
    fig, axes = figure(3, 2, height=2.1, sharex=True)
    for ax, house in zip(axes.ravel(), houses):
        rows = hourly[hourly["house"] == house]
        ax.bar(rows["hour"], rows["start_share_pct"], color=BLUE, width=0.72)
        ax.axvspan(15.5, 18.5, color=NEUTRAL, alpha=0.14, lw=0)
        ax.set_title(f"House {house} ({int(rows['cycle_starts'].sum())} cycles)", fontsize=9.5, color=INK, loc="left")
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylabel("Share of cycle starts (%)", fontsize=8, color=MUTED)
    for ax in axes[-1, :]:
        ax.set_xlabel("UK local hour of day (grey band: 16:00 to 19:00 peak)", fontsize=8, color=MUTED)
    return finish(fig, "When wash cycles start, by hour of day", path)


def weekday_distribution(weekday: pd.DataFrame, path: Path) -> Path:
    houses = sorted(weekday["house"].unique())
    fig, axes = figure(2, 3, height=2.5, sharex=True)
    for ax, house in zip(axes.ravel(), houses):
        rows = weekday[weekday["house"] == house].sort_values("weekday_number")
        ax.bar(rows["weekday"], rows["cycles_per_day"], color=BLUE, width=0.68)
        ax.set_title(f"House {house}", fontsize=9.5, color=INK, loc="left")
        ax.set_ylabel("Cycles per observed day", fontsize=8, color=MUTED)
        ax.set_xticks(range(7))
        ax.set_xticklabels(WEEKDAYS, fontsize=8)
    return finish(fig, "Wash cycles by day of week, adjusted for days observed", path)


def cross_household(summary: pd.DataFrame, path: Path) -> Path:
    labels = [f"House {h}\n({o} {'person' if o == 1 else 'people'})" for h, o in zip(summary["house"], summary["occupancy"])]
    measures = [("cycles_per_week", "Cycles per week"), ("median_duration_min", "Median cycle duration (min)"),
                ("median_energy_kwh", "Median energy per cycle (kWh)"), ("machine_kwh_per_year", "Machine energy (kWh a year)")]
    fig, axes = figure(2, 2, height=2.9)
    for ax, (column, title) in zip(axes.ravel(), measures):
        bars = ax.bar(labels, summary[column], color=BLUE, width=0.62)
        ax.bar_label(bars, fmt="%.3g", fontsize=8, color=INK, padding=2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.margins(y=0.15)
    return finish(fig, "Cross-household comparison of washing machine use", path)


def duration_energy(cycles: pd.DataFrame, path: Path) -> Path:
    full = cycles[cycles["complete"]].copy()
    full["House"] = "House " + full["house"].astype(str)
    order = [f"House {h}" for h in sorted(full["house"].unique())]
    fig, axes = figure(1, 2, height=4.4)
    box = dict(linewidth=0.8, fliersize=1.6, width=0.55, color=BLUE, order=order,
               flierprops=dict(markerfacecolor=MUTED, markeredgecolor=MUTED, alpha=0.5),
               medianprops=dict(color=INK, linewidth=1.3), boxprops=dict(alpha=0.75))
    sns.boxplot(full, x="House", y="duration_min", ax=axes[0], **box)
    sns.boxplot(full, x="House", y="energy_kwh", ax=axes[1], **box)
    axes[0].set_title("Cycle duration (minutes)", fontsize=9.5, color=INK, loc="left")
    axes[1].set_title("Energy per cycle (kWh)", fontsize=9.5, color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        style(ax)
    return finish(fig, "Cycle duration and energy per cycle, complete cycles", path)


def saving_by_household(opportunity: pd.DataFrame, path: Path) -> Path:
    labels = [f"House {h}" for h in opportunity["house"]]
    x = np.arange(len(labels))
    fig, axes = figure(1, 2, height=3.6)
    left, right = axes
    a = left.bar(x - 0.2, opportunity["machine_kwh_per_year"], width=0.38, color=BLUE, label="Washing machine energy now")
    b = left.bar(x + 0.2, opportunity["saving_cap_at_median_kwh_per_year"], width=0.38, color=ORANGE, label="Scenario saving: heating capped at own median")
    left.bar_label(a, fmt="%.0f", fontsize=7.5, color=INK, padding=1)
    left.bar_label(b, fmt="%.0f", fontsize=7.5, color=INK, padding=1)
    left.set_title("Energy a year (kWh)", fontsize=9.5, color=INK, loc="left")
    left.legend(frameon=False, fontsize=8, loc="upper left")
    left.margins(y=0.28)
    bars = right.bar(labels, opportunity["heat_share_of_cycle_energy_pct"], color=BLUE, width=0.6)
    right.bar_label(bars, fmt="%.0f%%", fontsize=8, color=INK, padding=1)
    right.set_title("Share of cycle energy used for heating (%)", fontsize=9.5, color=INK, loc="left")
    right.set_ylim(0, 105)
    left.set_xticks(x)
    left.set_xticklabels(labels, fontsize=8)
    return finish(fig, "Heating dominates wash energy, and each house's own cycles show room to reduce it", path)


def resolution(table: pd.DataFrame, cycle: pd.Series, house: int, experiments: pd.DataFrame, visibility: pd.DataFrame, path: Path) -> Path:
    fig, axes = figure(1, 3, height=3.6)
    ax = axes[0]
    start, end = int(cycle["start_pos"]) - 60, int(cycle["end_pos"]) + 60
    rows = table.iloc[max(start, 0):end + 1]
    minutes = np.arange(len(rows)) - 60
    aggregate = rows["aggregate_w"].astype("float64").reset_index(drop=True)
    ax.plot(minutes, aggregate, color=NEUTRAL, lw=0.8, label="Aggregate, 1 minute")
    ax.plot(minutes, rows["washing_machine_w"].astype("float64"), color=BLUE, lw=1.2, label="Washing machine, 1 minute")
    for step, colour in ((30, ORANGE),):
        blocks = aggregate.groupby(np.arange(len(aggregate)) // step).transform("mean")
        ax.plot(minutes, blocks, color=colour, lw=1.6, drawstyle="steps-post", label=f"Aggregate, {step} minutes")
    ax.set_title(f"House {house}: one wash, as a 30-minute meter sees it", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Minutes from cycle start", fontsize=8, color=MUTED)
    ax.set_ylabel("Watts", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5)
    pooled = experiments[experiments["house"] == "pooled"].set_index("experiment")
    names = [("primary", "1 min"), ("interval_15min", "15 min"), ("interval_30min", "30 min")]
    bars = axes[1].bar([label for _, label in names], [float(pooled.loc[key, "f1"]) for key, _ in names], color=BLUE, width=0.55)
    axes[1].bar_label(bars, fmt="%.2f", fontsize=8.5, color=INK, padding=1)
    axes[1].set_title("Held-out on/off F1, all houses pooled", fontsize=9, color=INK, loc="left")
    axes[1].set_ylim(0, 1)
    share = visibility.groupby("interval_minutes")["cycles_above_background_pct"].mean()
    bars = axes[2].bar([f"{k} min" for k in share.index], share.to_numpy(), color=BLUE, width=0.55)
    axes[2].bar_label(bars, fmt="%.0f%%", fontsize=8.5, color=INK, padding=1)
    axes[2].set_title("Cycles standing above normal household movement", fontsize=9, color=INK, loc="left")
    axes[2].set_ylim(0, 105)
    return finish(fig, "What coarser smart-meter readings lose", path)
