"""Household-level washing machine analysis on the prepared official 1-minute tables.

Primary results exclude frozen minutes. Hour and weekday results use the verified UK local
clock, and days without a usable offset are left out and counted.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.analysis.cycles import DEFAULT_PARAMS, CycleParams, detect_cycles

MINUTES_TO_KWH = 1 / 60000
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def primary_power(table: pd.DataFrame) -> np.ndarray:
    """Washing machine watts, with missing and frozen minutes as NaN."""
    return table["washing_machine_w"].where(table["valid"]).to_numpy(dtype="float64")


def quality_summary(table: pd.DataFrame, house: int) -> dict:
    grid = len(table)
    valid = table["valid"]
    present = ~table["missing"]
    return {
        "house": house,
        "first_minute": str(table["official_minute"].iloc[0]),
        "last_minute": str(table["official_minute"].iloc[-1]),
        "grid_minutes": grid,
        "missing_minutes": int(table["missing"].sum()),
        "missing_pct": round(float(table["missing"].mean() * 100), 2),
        "aggregate_frozen_minutes": int(table["aggregate_frozen"].sum()),
        "washing_machine_frozen_minutes": int(table["washing_machine_frozen"].sum()),
        "frozen_excluded_minutes": int((table["frozen"] & present).sum()),
        "frozen_excluded_pct_of_present": round(float((table["frozen"] & present).sum() / present.sum() * 100), 2),
        "valid_minutes": int(valid.sum()),
        "valid_days": round(float(valid.sum() / 1440), 1),
        "valid_pct_of_grid": round(float(valid.mean() * 100), 2),
        "issues_minutes_in_valid": int((table["issues_any"] & valid).sum()),
        "issues_pct_of_valid": round(float((table["issues_any"] & valid).sum() / valid.sum() * 100), 2),
        "aggregate_over_23kw_minutes": int((table["aggregate_over_23kw"] > 0).sum()),
        "median_readings_per_valid_minute": float(table.loc[valid, "n_readings"].median()),
        "valid_minutes_without_uk_local": int((table["uk_local_minute"].isna() & valid).sum()),
        "valid_pct_with_uk_local": round(float((table["uk_local_minute"].notna() & valid).sum() / valid.sum() * 100), 2),
    }


def power_states(table: pd.DataFrame, house: int, on_w: float = DEFAULT_PARAMS.on_w) -> dict:
    power = table.loc[table["valid"], "washing_machine_w"]
    aggregate = table.loc[table["valid"], "aggregate_w"]
    active = power > on_w
    return {
        "house": house,
        "zero_pct": round(float((power == 0).mean() * 100), 2),
        "standby_pct": round(float(((power > 0) & ~active).mean() * 100), 2),
        "active_pct": round(float(active.mean() * 100), 2),
        "active_hours_per_week": round(float(active.mean() * 168), 2),
        "mean_w": round(float(power.mean()), 2),
        "mean_w_when_active": round(float(power[active].mean()), 1),
        "machine_kwh_per_day": round(float(power.sum() * MINUTES_TO_KWH / (len(power) / 1440)), 3),
        "aggregate_kwh_per_day": round(float(aggregate.sum() * MINUTES_TO_KWH / (len(power) / 1440)), 2),
        "machine_share_of_aggregate_energy_pct": round(float(power.sum() / aggregate.sum() * 100), 2),
    }


def household_cycles(table: pd.DataFrame, house: int, params: CycleParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Detected cycles with original and UK local start times."""
    cycles = detect_cycles(primary_power(table), params)
    cycles.insert(0, "house", house)
    if cycles.empty:
        return cycles
    starts = cycles["start_pos"].to_numpy()
    cycles["start_official"] = table["official_minute"].to_numpy()[starts]
    cycles["end_official"] = table["official_minute"].to_numpy()[cycles["end_pos"].to_numpy()]
    local = table["uk_local_minute"].to_numpy()[starts]
    cycles["start_uk_local"] = local
    cycles["start_hour"] = pd.Series(local).dt.hour.astype("Int64").to_numpy()
    cycles["start_weekday"] = pd.Series(local).dt.day_of_week.astype("Int64").to_numpy()
    cycles["heat_energy_share"] = (cycles["heat_energy_kwh"] / cycles["energy_kwh"]).round(4)
    return cycles


def cycle_summary(cycles: pd.DataFrame, table: pd.DataFrame, house: int) -> dict:
    """Per-household figures. Medians use complete cycles, so outages can't shorten them."""
    valid_days = float(table["valid"].sum() / 1440)
    power = table.loc[table["valid"], "washing_machine_w"]
    full = cycles[cycles["complete"]]
    return {
        "house": house,
        "cycles": len(cycles),
        "complete_cycles": len(full),
        "cycles_per_week": round(len(cycles) / (valid_days / 7), 2),
        "median_duration_min": float(full["duration_min"].median()),
        "p10_duration_min": float(full["duration_min"].quantile(0.1)),
        "p90_duration_min": float(full["duration_min"].quantile(0.9)),
        "median_peak_w": round(float(full["peak_w"].median()), 0),
        "median_energy_kwh": round(float(full["energy_kwh"].median()), 3),
        "p10_energy_kwh": round(float(full["energy_kwh"].quantile(0.1)), 3),
        "p90_energy_kwh": round(float(full["energy_kwh"].quantile(0.9)), 3),
        "heated_cycles_pct": round(float(full["heated"].mean() * 100), 1),
        "heat_share_of_cycle_energy_pct": round(float(full["heat_energy_kwh"].sum() / full["energy_kwh"].sum() * 100), 1),
        "cycles_over_240_min": int((cycles["duration_min"] > 240).sum()),
        "energy_in_detected_cycles_pct": round(float(cycles["energy_kwh"].sum() / (power.sum() * MINUTES_TO_KWH) * 100), 1),
        "machine_kwh_per_year": round(float(power.sum() * MINUTES_TO_KWH / valid_days * 365), 1),
        "valid_days": round(valid_days, 1),
    }


def _energy_by(table: pd.DataFrame, key: pd.Series, categories, on_w: float) -> pd.Series:
    power = table["washing_machine_w"].where(table["valid"] & (table["washing_machine_w"] > on_w)).astype("float64")
    return (power.groupby(key.to_numpy()).sum() * MINUTES_TO_KWH).reindex(categories, fill_value=0.0)


def hourly_usage(cycles: pd.DataFrame, table: pd.DataFrame, house: int, on_w: float = DEFAULT_PARAMS.on_w) -> pd.DataFrame:
    """Cycle starts and machine energy by UK local hour, on days with a usable clock."""
    usable = table["uk_local_minute"].notna()
    starts = cycles["start_hour"].dropna().astype(int).value_counts().reindex(range(24), fill_value=0)
    local = table.loc[usable]
    energy = _energy_by(local, local["uk_local_minute"].dt.hour, range(24), on_w)
    available = (local["valid"].groupby(local["uk_local_minute"].dt.hour.to_numpy()).sum() / 60).reindex(range(24), fill_value=0)
    return pd.DataFrame({
        "house": house,
        "hour": range(24),
        "cycle_starts": starts.to_numpy(),
        "start_share_pct": (starts / starts.sum() * 100).round(2).to_numpy(),
        "energy_kwh": energy.round(3).to_numpy(),
        "energy_share_pct": (energy / energy.sum() * 100).round(2).to_numpy(),
        "valid_hours_observed": available.round(1).to_numpy(),
    })


def weekday_usage(cycles: pd.DataFrame, table: pd.DataFrame, house: int, on_w: float = DEFAULT_PARAMS.on_w) -> pd.DataFrame:
    """Cycle starts and machine energy by UK local weekday, normalised for data coverage."""
    usable = table["uk_local_minute"].notna()
    local = table.loc[usable]
    weekday = local["uk_local_minute"].dt.day_of_week
    starts = cycles["start_weekday"].dropna().astype(int).value_counts().reindex(range(7), fill_value=0)
    days = (local["valid"].groupby(weekday.to_numpy()).sum() / 1440).reindex(range(7), fill_value=0)
    energy = _energy_by(local, weekday, range(7), on_w)
    return pd.DataFrame({
        "house": house,
        "weekday": WEEKDAYS,
        "weekday_number": range(7),
        "cycle_starts": starts.to_numpy(),
        "valid_days_observed": days.round(1).to_numpy(),
        "cycles_per_day": (starts / days.replace(0, np.nan)).round(3).to_numpy(),
        "start_share_pct": (starts / starts.sum() * 100).round(2).to_numpy(),
        "energy_kwh": energy.round(3).to_numpy(),
    })


def clock_sensitivity(cycles: pd.DataFrame, table: pd.DataFrame, house: int) -> dict:
    """How wrong would hour-of-day results be without the clock correction?"""
    usable = cycles["start_uk_local"].notna()
    corrected = cycles.loc[usable, "start_hour"].astype(int).to_numpy()
    uncorrected = pd.to_datetime(cycles.loc[usable, "start_official"]).dt.hour.to_numpy()
    evening = lambda hours: float(((hours >= 16) & (hours < 19)).mean() * 100) if len(hours) else float("nan")
    return {
        "house": house,
        "cycles": len(cycles),
        "cycles_without_usable_clock": int((~usable).sum()),
        "cycles_with_hour_changed_pct": round(float((corrected != uncorrected).mean() * 100), 1) if len(corrected) else float("nan"),
        "evening_share_uncorrected_pct": round(evening(uncorrected), 1),
        "evening_share_corrected_pct": round(evening(corrected), 1),
    }


def frozen_sensitivity(table: pd.DataFrame, house: int, params: CycleParams = DEFAULT_PARAMS) -> dict:
    """Compare results with frozen minutes excluded (primary) and included."""
    included = table["washing_machine_w"].where(~table["missing"]).to_numpy(dtype="float64")
    with_frozen = detect_cycles(included, params)
    primary = detect_cycles(primary_power(table), params)
    present = ~table["missing"]
    power_all = table.loc[present, "washing_machine_w"]
    power_primary = table.loc[table["valid"], "washing_machine_w"]
    return {
        "house": house,
        "cycles_excluding_frozen": len(primary),
        "cycles_including_frozen": len(with_frozen),
        "kwh_per_day_excluding_frozen": round(float(power_primary.sum() * MINUTES_TO_KWH / (len(power_primary) / 1440)), 3),
        "kwh_per_day_including_frozen": round(float(power_all.sum() * MINUTES_TO_KWH / (len(power_all) / 1440)), 3),
        "active_pct_excluding_frozen": round(float((power_primary > params.on_w).mean() * 100), 2),
        "active_pct_including_frozen": round(float((power_all > params.on_w).mean() * 100), 2),
        "median_kwh_excluding_frozen": round(float(primary["energy_kwh"].median()), 3),
        "median_kwh_including_frozen": round(float(with_frozen["energy_kwh"].median()), 3),
    }


def parameter_sensitivity(table: pd.DataFrame, house: int) -> pd.DataFrame:
    """How much do cycle results depend on each detector setting?"""
    power = primary_power(table)
    base = DEFAULT_PARAMS
    settings = [("default", base)]
    settings += [(f"on_w={v:g}", replace(base, on_w=v)) for v in (10, 50)]
    settings += [(f"max_gap_min={v}", replace(base, max_gap_min=v)) for v in (5, 15, 30)]
    settings += [(f"min_duration_min={v}", replace(base, min_duration_min=v)) for v in (10, 30)]
    settings += [(f"min_peak_w={v:g}", replace(base, min_peak_w=v)) for v in (100, 500)]
    total = np.nansum(power[power > base.on_w])
    rows = []
    for label, params in settings:
        cycles = detect_cycles(power, params)
        rows.append({"house": house, "setting": label, "cycles": len(cycles),
                     "median_duration_min": float(cycles["duration_min"].median()),
                     "median_energy_kwh": round(float(cycles["energy_kwh"].median()), 3),
                     "energy_captured_pct": round(float(cycles["energy_kwh"].sum() * 60000 / total * 100), 1)})
    return pd.DataFrame(rows)


def representative_day(cycles: pd.DataFrame, table: pd.DataFrame) -> pd.Timestamp | None:
    """A UK local day with full coverage and one complete cycle of typical duration."""
    local = table["uk_local_minute"]
    days = local.dt.normalize()
    coverage = table["valid"].groupby(days.to_numpy()).sum()
    full_days = set(coverage[coverage >= 1430].index)
    full = cycles[cycles["complete"] & cycles["start_uk_local"].notna()].copy()
    if full.empty:
        return None
    full["day"] = pd.to_datetime(full["start_uk_local"]).dt.normalize()
    per_day = full.groupby("day").size()
    candidates = full[full["day"].isin(full_days) & full["day"].map(per_day).eq(1)].copy()
    if candidates.empty:
        return None
    target = cycles.loc[cycles["complete"], "duration_min"].median()
    candidates["distance"] = (candidates["duration_min"] - target).abs()
    return candidates.sort_values(["distance", "day"])["day"].iloc[0]


def example_cycles(cycles: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Complete cycles spread across the duration range, chosen deterministically."""
    full = cycles[cycles["complete"]].sort_values(["duration_min", "start_pos"]).reset_index(drop=True)
    if len(full) <= n:
        return full
    return full.iloc[[int(q * (len(full) - 1)) for q in np.linspace(0.1, 0.9, n)]]
