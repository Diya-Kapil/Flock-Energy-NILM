"""Before-and-after plots for the House 1 cleaning."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import HOUSE1_LABELS, SENSORS  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e5e2", "#fcfcfb"
BEFORE, AFTER = "#eb6834", "#2a78d6"  # Passed the dataviz palette validator on the light surface.
PLOT_SENSORS = ["Aggregate", "Appliance5", "Appliance1", "Appliance4"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _power_axis(ax, top: float) -> None:
    ax.set_yscale("symlog", linthresh=100)
    ax.set_ylim(-5, top)
    ax.set_yticks([t for t in (0, 100, 1000, 10000, 100000) if t <= top])
    ax.set_yticklabels([f"{t:,}" for t in (0, 100, 1000, 10000, 100000) if t <= top])
    ax.set_ylabel("Watts (log scale above 100)", fontsize=8, color=MUTED)


def daily_max_before(rows: pd.DataFrame, sensor: str) -> pd.Series:
    stamps = pd.to_datetime(rows["Unix"].to_numpy(), unit="s")
    frame = pd.Series(rows[sensor].to_numpy(), index=stamps)
    days = frame.groupby(frame.index.normalize()).max()
    # Days without data must stay empty, or the line would draw straight across an outage.
    return days.reindex(pd.date_range(days.index.min(), days.index.max(), freq="D"))


def daily_max_after(minutes: pd.DataFrame, sensor: str) -> pd.Series:
    frame = pd.Series(minutes[sensor].to_numpy(), index=minutes["raw_clock_minute"].to_numpy())
    return frame.groupby(pd.DatetimeIndex(frame.index).normalize()).max()


def pick_zoom(minutes: pd.DataFrame, sensor: str) -> pd.Timestamp:
    """Choose a window that shows a removed or flagged reading next to normal activity."""
    flag = "Aggregate_over_23kw" if sensor == "Aggregate" else f"{sensor}_spikes"
    frame = minutes.loc[(minutes["source_part"] == 2) & (minutes[flag] > 0), ["raw_clock_minute", sensor]]
    if sensor == "Aggregate":
        calm = frame[frame[sensor] < 8000]
        frame = calm if len(calm) else frame
    if frame.empty:
        return minutes["raw_clock_minute"].iloc[len(minutes) // 2]
    days = frame["raw_clock_minute"].dt.normalize()
    activity = minutes[["raw_clock_minute", sensor]].copy()
    activity["day"] = activity["raw_clock_minute"].dt.normalize()
    busy = (activity[sensor] > 500).groupby(activity["day"]).sum()
    best_day = busy.reindex(days.unique()).idxmax()
    return frame.loc[days == best_day, "raw_clock_minute"].iloc[0]


def before_after(part1: pd.DataFrame, part2: pd.DataFrame, minutes: pd.DataFrame, sensor: str, out: Path,
                 switch: pd.Timestamp) -> Path:
    label = HOUSE1_LABELS[sensor]
    raw = pd.concat([part1[["Unix", sensor]], part2[["Unix", sensor]]], ignore_index=True)
    centre = pick_zoom(minutes, sensor)
    lo, hi = centre - pd.Timedelta(hours=1), centre + pd.Timedelta(hours=5)
    zoom_raw = part2[(part2["Unix"] >= lo.value // 10**9) & (part2["Unix"] < hi.value // 10**9)][["Unix", sensor]].dropna().sort_values("Unix")
    zoom_stamps = pd.to_datetime(zoom_raw["Unix"].to_numpy(), unit="s")
    zoom_after = minutes[(minutes["raw_clock_minute"] >= lo) & (minutes["raw_clock_minute"] < hi)]

    before_days = daily_max_before(raw, sensor)
    after_days = daily_max_after(minutes, sensor)
    top = max(float(np.nanmax(before_days.to_numpy())), float(np.nanmax(zoom_raw[sensor].to_numpy())) if len(zoom_raw) else 0) * 1.6

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.4), constrained_layout=True, sharey=True,
                             gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0, 0], before_days.index, before_days.to_numpy(), BEFORE, "Before: raw files, highest reading each day"),
        (axes[1, 0], after_days.index, after_days.to_numpy(), AFTER, "After: cleaned 1-minute table, highest value each day"),
        (axes[0, 1], zoom_stamps, zoom_raw[sensor].to_numpy(), BEFORE, f"Before: raw readings, {centre:%d %b %Y}"),
        (axes[1, 1], zoom_after["raw_clock_minute"], zoom_after[sensor].to_numpy(dtype="float64"), AFTER, f"After: 1-minute values, {centre:%d %b %Y}"),
    ]
    for ax, x, y, colour, title in panels:
        _style(ax)
        ax.plot(x, y, color=colour, lw=1.0)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _power_axis(ax, top)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.axvline(switch, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.annotate("Part 2 used from here", (switch, 0.995), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=MUTED, va="top")
        ax.set_xlim(before_days.index.min(), before_days.index.max())
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    fig.suptitle(f"House 1, {label}: before and after cleaning", fontsize=12, color=INK, x=0.01, ha="left")
    path = out / f"before_after_{sensor.lower()}.png"
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def coverage(minutes: pd.DataFrame, out: Path, switch: pd.Timestamp) -> Path:
    days = minutes["raw_clock_minute"].dt.normalize()
    table = pd.DataFrame({sensor: (~minutes[f"{sensor}_missing"]).groupby(days.to_numpy()).mean() * 100 for sensor in SENSORS})
    table = table.reindex(pd.date_range(table.index.min(), table.index.max(), freq="D"))
    ramp = LinearSegmentedColormap.from_list("coverage", ["#f1f0ec", "#a9c8ee", "#2a78d6", "#123f7a"])
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    start, end = matplotlib.dates.date2num(table.index[0]), matplotlib.dates.date2num(table.index[-1])
    image = ax.imshow(table.T.to_numpy(), aspect="auto", cmap=ramp, vmin=0, vmax=100, interpolation="nearest",
                      extent=(start, end + 1, len(SENSORS) - 0.5, -0.5))
    ax.xaxis_date()
    ax.set_yticks(range(len(SENSORS)))
    ax.set_yticklabels([HOUSE1_LABELS[s] for s in SENSORS], fontsize=8.5, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(matplotlib.dates.date2num(switch), color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("Part 2 used from here", (matplotlib.dates.date2num(switch), -0.5), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=INK, annotation_clip=False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    bar.set_label("Minutes with a reading (%)", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    ax.set_title("House 1 cleaned table: share of minutes with a reading, by sensor and day", fontsize=11, color=INK, loc="left", pad=14)
    path = out / "coverage_by_sensor.png"
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_plots(part1: pd.DataFrame, part2: pd.DataFrame, minutes: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    switch = minutes.loc[minutes["source_part"] == 2, "raw_clock_minute"].iloc[0]
    paths = [before_after(part1, part2, minutes, sensor, out, switch) for sensor in PLOT_SENSORS]
    paths.append(coverage(minutes, out, switch))
    return paths
