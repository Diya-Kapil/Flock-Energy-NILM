"""Evaluate held-out predictions: per house, pooled, with and without Issues, and failures."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.analysis.cycles import DEFAULT_PARAMS, detect_cycles
from src.models.baselines import always_zero
from src.models.metrics import ON_W, evaluate, match_cycles

PEAK_HOURS = (16, 19)
SUMMARY_COLUMNS = ["house", "model", "subset", "minutes", "active_minutes", "mae_w", "rmse_w", "r2", "mae_active_w",
                   "precision", "recall", "f1", "actual_kwh", "predicted_kwh", "energy_error_pct"]


def predictions_on_grid(frame: pd.DataFrame, grid_length: int, column: str) -> np.ndarray:
    """Place a prediction column on the house's full grid, with NaN elsewhere."""
    series = np.full(grid_length, np.nan)
    series[frame["grid_pos"].to_numpy()] = frame[column].to_numpy(dtype="float64")
    return series


def _rows_for(frame: pd.DataFrame, house, subset: str) -> list[dict]:
    y, p = frame["actual_w"].to_numpy(), frame["predicted_w"].to_numpy()
    threshold = float(frame["threshold_w"].iloc[0])
    model = {"house": house, "model": "random_forest", "subset": subset, **evaluate(y, p, threshold)}
    baseline = {"house": house, "model": "always_zero", "subset": subset, **evaluate(y, always_zero(len(y)), threshold)}
    return [model, baseline]


def metric_tables(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Metrics per house and pooled, on all valid minutes and excluding Issues = 1."""
    rows = []
    for subset, picker in (("all_valid", lambda f: f), ("excluding_issues", lambda f: f[~f["issues_any"]])):
        for house, frame in predictions.items():
            rows += _rows_for(picker(frame), house, subset)
        pooled = pd.concat([picker(f) for f in predictions.values()], ignore_index=True)
        # Each row keeps the threshold chosen for its own fold, so pooled on/off is computed per row.
        actual, predicted = pooled["actual_w"].to_numpy(), pooled["predicted_w"].to_numpy()
        model = evaluate(actual, predicted, 0.0)
        model.update(_pooled_on_off(pooled))
        rows.append({"house": "pooled", "model": "random_forest", "subset": subset, **model})
        rows.append({"house": "pooled", "model": "always_zero", "subset": subset, **evaluate(actual, always_zero(len(actual)), 0.0)})
    return pd.DataFrame(rows)


def _pooled_on_off(pooled: pd.DataFrame) -> dict:
    from src.models.metrics import on_off
    return on_off(pooled["actual_w"].to_numpy() > ON_W, pooled["predicted_w"].to_numpy() > pooled["threshold_w"].to_numpy())


def macro_means(metrics: pd.DataFrame) -> pd.DataFrame:
    """Unweighted mean across houses, so one large house can't dominate."""
    numeric = ["mae_w", "rmse_w", "r2", "mae_active_w", "precision", "recall", "f1", "energy_error_pct"]
    houses = metrics[metrics["house"] != "pooled"]
    return houses.groupby(["model", "subset"])[numeric].mean().reset_index()


def predicted_cycles(series: np.ndarray, threshold_w: float) -> pd.DataFrame:
    """Cycles in a predicted series, using the same gap and duration rules as actual cycles.

    The peak test is relaxed to the threshold, because averaged predictions rarely reach
    the true peak.
    """
    params = replace(DEFAULT_PARAMS, on_w=threshold_w, min_peak_w=threshold_w)
    return detect_cycles(series, params)


def cycle_performance(predictions: dict[int, pd.DataFrame], grid_lengths: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cycle detection per house, plus a row per actual cycle saying whether it was found."""
    rows, details = [], []
    for house, frame in predictions.items():
        threshold = float(frame["threshold_w"].iloc[0])
        actual = detect_cycles(predictions_on_grid(frame, grid_lengths[house], "actual_w"))
        predicted = predicted_cycles(predictions_on_grid(frame, grid_lengths[house], "predicted_w"), threshold)
        result = match_cycles(actual, predicted)
        flags = result.pop("matched_flags")
        result.pop("matched_prediction_positions")
        rows.append({"house": house, **result})
        details.append(actual.assign(house=house, detected=flags))
    table = pd.DataFrame(rows)
    totals = table[["actual_cycles", "predicted_cycles", "matched"]].sum()
    precision = totals["matched"] / totals["predicted_cycles"] if totals["predicted_cycles"] else 0.0
    recall = totals["matched"] / totals["actual_cycles"] if totals["actual_cycles"] else 0.0
    pooled = {"house": "pooled", **totals.to_dict(), "cycle_precision": precision, "cycle_recall": recall,
              "cycle_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return pd.concat([table, pd.DataFrame([pooled])], ignore_index=True), pd.concat(details, ignore_index=True)


def missed_cycle_analysis(details: pd.DataFrame) -> pd.DataFrame:
    """Which kinds of cycle does the model find, by heating and duration?"""
    rows = []
    def add(group: str, label: str, part: pd.DataFrame) -> None:
        if len(part):
            rows.append({"group": group, "label": label, "cycles": len(part), "detected": int(part["detected"].sum()),
                         "recall": round(float(part["detected"].mean()), 3), "median_energy_kwh": round(float(part["energy_kwh"].median()), 3)})
    add("heating", "heated", details[details["heated"]])
    add("heating", "unheated", details[~details["heated"]])
    for label, lo, hi in (("under 45 min", 0, 45), ("45 to 89 min", 45, 90), ("90 min or more", 90, 10**6)):
        add("duration", label, details[(details["duration_min"] >= lo) & (details["duration_min"] < hi)])
    for label, lo, hi in (("under 5 heating min", 0, 5), ("5 to 14 heating min", 5, 15), ("15 or more heating min", 15, 10**6)):
        add("heating length", label, details[(details["heat_minutes"] >= lo) & (details["heat_minutes"] < hi)])
    return pd.DataFrame(rows)


def false_positive_sources(predictions: dict[int, pd.DataFrame], tables: dict[int, pd.DataFrame], names: dict[int, dict[str, str]]) -> pd.DataFrame:
    """During false alarms, which other metered appliance was running, if any?"""
    rows = []
    for house, frame in predictions.items():
        threshold = float(frame["threshold_w"].iloc[0])
        wrong = frame[(frame["actual_w"] <= ON_W) & (frame["predicted_w"] > threshold)]
        others = [c for c in tables[house].columns if c.startswith("other_")]
        readings = tables[house][others].iloc[wrong["grid_pos"].to_numpy()].fillna(0.0)
        strongest = readings.idxmax(axis=1).where(readings.max(axis=1) >= 100, "none")
        energy = pd.Series(wrong["predicted_w"].to_numpy(), index=strongest.index).groupby(strongest).sum()
        for source, value in energy.sort_values(ascending=False).items():
            label = "No metered appliance at 100 W or more" if source == "none" else names[house][source]
            rows.append({"house": house, "running_appliance": label, "false_alarm_minutes": int((strongest == source).sum()),
                         "share_of_false_alarm_energy_pct": round(float(value / energy.sum() * 100), 1)})
    return pd.DataFrame(rows)


def false_alarm_episodes(frame: pd.DataFrame, grid_length: int, n: int = 3) -> pd.DataFrame:
    """Predicted cycles with no overlapping actual operation, largest predicted energy first."""
    threshold = float(frame["threshold_w"].iloc[0])
    predicted = predicted_cycles(predictions_on_grid(frame, grid_length, "predicted_w"), threshold)
    if predicted.empty:
        return predicted
    actual = predictions_on_grid(frame, grid_length, "actual_w")
    actual_kwh = [float(np.nansum(actual[s:e + 1]) / 60000) for s, e in zip(predicted["start_pos"], predicted["end_pos"])]
    predicted = predicted.assign(actual_kwh=actual_kwh)
    return predicted[predicted["actual_kwh"] < 0.02].sort_values("energy_kwh", ascending=False).head(n)


def peak_share(frame: pd.DataFrame, column: str) -> float:
    """Share of a series' energy falling in the 16:00 to 19:00 UK local peak."""
    usable = frame[frame["uk_local_minute"].notna()]
    hours = pd.to_datetime(usable["uk_local_minute"]).dt.hour
    values = usable[column].to_numpy(dtype="float64")
    in_peak = (hours >= PEAK_HOURS[0]) & (hours < PEAK_HOURS[1])
    return float(values[in_peak.to_numpy()].sum() / values.sum() * 100) if values.sum() else float("nan")


def practical_estimates(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Could the model support the household-level numbers a utility would actually use?"""
    rows = []
    for house, frame in predictions.items():
        days = len(frame) / 1440
        actual, predicted = frame["actual_w"].sum() / 60000, frame["predicted_w"].sum() / 60000
        rows.append({"house": house, "actual_kwh_per_year": round(float(actual / days * 365), 1),
                     "predicted_kwh_per_year": round(float(predicted / days * 365), 1),
                     "actual_peak_share_pct": round(peak_share(frame, "actual_w"), 1),
                     "predicted_peak_share_pct": round(peak_share(frame, "predicted_w"), 1)})
    return pd.DataFrame(rows)


def error_sources(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Split absolute error by whether the machine was really off or operating."""
    rows = []
    frames = {**predictions, "pooled": pd.concat(predictions.values(), ignore_index=True)}
    for house, frame in frames.items():
        active = frame["actual_w"].to_numpy() > ON_W
        error = np.abs(frame["predicted_w"].to_numpy() - frame["actual_w"].to_numpy())
        total = float(error.sum())
        gated = np.where(frame["predicted_w"].to_numpy() > frame["threshold_w"].to_numpy(), frame["predicted_w"].to_numpy(), 0.0)
        rows.append({
            "house": house, "active_minutes_pct": round(float(active.mean() * 100), 2),
            "error_from_off_minutes_pct": round(float(error[~active].sum() / total * 100), 1),
            "mean_prediction_when_off_w": round(float(frame["predicted_w"].to_numpy()[~active].mean()), 1),
            "mean_prediction_when_active_w": round(float(frame["predicted_w"].to_numpy()[active].mean()), 1),
            "mean_actual_when_active_w": round(float(frame["actual_w"].to_numpy()[active].mean()), 1),
            "off_minutes_predicted_on_pct": round(float((frame["predicted_w"].to_numpy()[~active] > frame["threshold_w"].to_numpy()[~active]).mean() * 100), 2),
            "mae_w": round(float(error.mean()), 2),
            "mae_if_below_threshold_set_to_zero_w": round(float(np.abs(gated - frame["actual_w"].to_numpy()).mean()), 2),
            "always_zero_mae_w": round(float(frame["actual_w"].mean()), 2),
        })
    return pd.DataFrame(rows)


def error_by_aggregate_level(predictions: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Where false load appears, by how busy the rest of the house is."""
    pooled = pd.concat(predictions.values(), ignore_index=True)
    bands = [(0, 500, "under 500 W"), (500, 1500, "500 to 1,499 W"), (1500, 3000, "1,500 to 2,999 W"), (3000, 6000, "3,000 to 5,999 W"), (6000, 10**9, "6,000 W or more")]
    rows = []
    for lo, hi, label in bands:
        part = pooled[(pooled["aggregate_w"] >= lo) & (pooled["aggregate_w"] < hi)]
        active = part["actual_w"].to_numpy() > ON_W
        rows.append({"aggregate_level": label, "minutes": len(part), "machine_active_pct": round(float(active.mean() * 100), 1),
                     "mean_actual_w": round(float(part["actual_w"].mean()), 1), "mean_predicted_w": round(float(part["predicted_w"].mean()), 1),
                     "mae_w": round(float(np.abs(part["predicted_w"] - part["actual_w"]).mean()), 1)})
    return pd.DataFrame(rows)


def typical_cycle(details: pd.DataFrame, house: int) -> pd.Series:
    """The complete cycle closest to this house's median duration, for plotting."""
    own = details[(details["house"] == house) & details["complete"]]
    distance = (own["duration_min"] - own["duration_min"].median()).abs()
    return own.assign(distance=distance).sort_values(["distance", "start_pos"]).iloc[0]


def failure_cases(predictions: dict[int, pd.DataFrame], grids: dict[int, int], details: pd.DataFrame, tables: dict[int, pd.DataFrame],
                  names: dict[int, dict[str, str]], per_kind: int = 3) -> list[dict]:
    """The largest false alarms and missed cycles, taking at most one of each per house."""
    alarms, misses = [], []
    for house, frame in predictions.items():
        episodes = false_alarm_episodes(frame, grids[house], 1)
        for _, episode in episodes.iterrows():
            start, end = int(episode["start_pos"]), int(episode["end_pos"])
            others = [c for c in tables[house].columns if c.startswith("other_")]
            energy = tables[house][others].iloc[start:end + 1].fillna(0).sum()
            top = energy.idxmax()
            cause = names[house][top] if energy.max() / 60000 >= 0.05 else "no metered appliance"
            alarms.append({"house": house, "kind": "false alarm", "start_pos": start, "end_pos": end, "energy_kwh": float(episode["energy_kwh"]),
                           "running_appliance": cause, "title": f"false alarm, {episode['energy_kwh']:.2f} kWh predicted, largest other load: {cause}"})
        missed = details[(details["house"] == house) & ~details["detected"] & details["complete"]].sort_values("energy_kwh", ascending=False).head(1)
        for _, cycle in missed.iterrows():
            misses.append({"house": house, "kind": "missed cycle", "start_pos": int(cycle["start_pos"]), "end_pos": int(cycle["end_pos"]), "energy_kwh": float(cycle["energy_kwh"]),
                           "running_appliance": "", "title": f"missed cycle, {cycle['duration_min']:.0f} min, {cycle['energy_kwh']:.2f} kWh actual"})
    alarms.sort(key=lambda c: -c["energy_kwh"])
    misses.sort(key=lambda c: -c["energy_kwh"])
    return alarms[:per_kind] + misses[:per_kind]
