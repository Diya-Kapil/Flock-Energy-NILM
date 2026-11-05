"""Charts for model evaluation. Actual is blue and predicted is orange throughout."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.analysis.plots import BLUE, GRID, INK, MUTED, NEUTRAL, ORANGE, SURFACE, figure, finish, style  # noqa: E402

RAMP = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281", "#0d366b"])


def _window(frame: pd.DataFrame, start: int, end: int, grid_length: int, margin: int = 90) -> pd.DataFrame:
    lo, hi = max(start - margin, 0), min(end + margin, grid_length - 1)
    return frame[(frame["grid_pos"] >= lo) & (frame["grid_pos"] <= hi)]


def _trace(ax, rows: pd.DataFrame, title: str, start: int) -> None:
    x = rows["grid_pos"].to_numpy() - start
    ax.plot(x, rows["aggregate_w"], color=NEUTRAL, lw=0.8, label="Household aggregate (model input)")
    ax.plot(x, rows["actual_w"], color=BLUE, lw=1.3, label="Actual washing machine")
    ax.plot(x, rows["predicted_w"], color=ORANGE, lw=1.3, label="Predicted washing machine")
    ax.set_title(title, fontsize=8.6, color=INK, loc="left")
    ax.set_ylabel("Watts", fontsize=8, color=MUTED)
    ax.set_ylim(bottom=0)


def actual_vs_predicted(chosen: dict[int, pd.Series], predictions: dict[int, pd.DataFrame], grids: dict[int, int], path: Path) -> Path:
    fig, axes = figure(3, 2, height=2.5)
    for ax, (house, cycle) in zip(axes.ravel(), chosen.items()):
        start, end = int(cycle["start_pos"]), int(cycle["end_pos"])
        rows = _window(predictions[house], start, end, grids[house])
        _trace(ax, rows, f"House {house} held out: typical cycle, {cycle['duration_min']:.0f} min, {cycle['energy_kwh']:.2f} kWh, "
                         f"{'found' if cycle['detected'] else 'missed'}", start)
        ax.set_xlabel("Minutes from cycle start", fontsize=8, color=MUTED)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.5, ncol=3)
    return finish(fig, "Actual and predicted power", path)


def scatter_density(predictions: dict[int, pd.DataFrame], path: Path) -> Path:
    fig, axes = figure(2, 3, height=3.1)
    for ax, (house, frame) in zip(axes.ravel(), predictions.items()):
        counts = ax.hexbin(frame["actual_w"], frame["predicted_w"], gridsize=45, extent=(0, 3000, 0, 3000), bins="log", mincnt=1, cmap=RAMP, linewidths=0)
        ax.plot([0, 3000], [0, 3000], color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.set_title(f"House {house}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Actual (W)", fontsize=8, color=MUTED)
        ax.set_ylabel("Predicted (W)", fontsize=8, color=MUTED)
    bar = fig.colorbar(counts, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    bar.set_label("Minutes (log scale)", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    bar.outline.set_visible(False)
    return finish(fig, "Predicted against actual power, each house held out. A perfect model sits on the dashed line", path)


def model_vs_baseline(metrics: pd.DataFrame, path: Path) -> Path:
    table = metrics[metrics["subset"] == "all_valid"].copy()
    order = [str(h) for h in table["house"].unique()]
    table["house"] = table["house"].astype(str)
    measures = [("mae_w", "MAE, all valid minutes (W)"), ("rmse_w", "RMSE, all valid minutes (W)"), ("mae_active_w", "MAE while operating (W)"),
                ("r2", "R²"), ("f1", "On/off F1"), ("energy_error_pct", "Energy error (%)")]
    fig, axes = figure(3, 2, height=2.45)
    x = np.arange(len(order))
    for ax, (column, title) in zip(axes.ravel(), measures):
        model = table[table["model"] == "random_forest"].set_index("house").reindex(order)[column]
        base = table[table["model"] == "always_zero"].set_index("house").reindex(order)[column]
        a = ax.bar(x - 0.2, model, width=0.38, color=ORANGE, label="Random Forest")
        b = ax.bar(x + 0.2, base, width=0.38, color=BLUE, label="Always zero")
        ax.bar_label(a, fmt="%.3g", fontsize=6.5, color=INK, padding=1)
        ax.bar_label(b, fmt="%.3g", fontsize=6.5, color=INK, padding=1)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([("Pooled" if h == "pooled" else f"House {h}") for h in order], fontsize=7.8)
        ax.set_title(title, fontsize=9.3, color=INK, loc="left")
        ax.margins(y=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncol=2)
    return finish(fig, "Random Forest against the always-zero baseline, held-out houses", path)


def feature_importance(importance: pd.Series, path: Path, top: int = 15) -> Path:
    fig, axes = figure(1, 1, width=8, height=4.6)
    ax = axes[0]
    rows = importance.sort_values(ascending=False).head(top)[::-1]
    bars = ax.barh(rows.index, rows.to_numpy(), color=BLUE, height=0.65)
    ax.bar_label(bars, fmt="%.3f", fontsize=7.5, color=INK, padding=2)
    ax.tick_params(axis="y", labelsize=8, colors=INK)
    ax.set_xlabel("Mean impurity importance across folds", fontsize=8, color=MUTED)
    ax.margins(x=0.1)
    return finish(fig, f"Top {top} aggregate features", path)


def failure_cases(cases: list[dict], predictions: dict[int, pd.DataFrame], grids: dict[int, int], path: Path) -> Path:
    rows_needed = int(np.ceil(len(cases) / 2))
    fig, axes = figure(rows_needed, 2, height=2.4)
    for ax in axes.ravel()[len(cases):]:
        ax.set_axis_off()
    for ax, case in zip(axes.ravel(), cases):
        house, start, end = case["house"], int(case["start_pos"]), int(case["end_pos"])
        _trace(ax, _window(predictions[house], start, end, grids[house]), f"House {house}: {case['title']}", start)
        ax.set_xlabel("Minutes from event start", fontsize=8, color=MUTED)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.5, ncol=3)
    return finish(fig, "Failure cases: the model's largest false alarms and missed cycles", path)
