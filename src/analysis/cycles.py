"""Detect washing machine cycles in a 1-minute power series.

The definition is deliberately simple, and its values are justified from the data in
RESULTS.md: power above a threshold, off-gaps up to a tolerance bridged, then a minimum
duration and peak. Missing minutes are never treated as "off": a gap containing one
splits the run and marks the cycles beside it incomplete.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CycleParams:
    on_w: float = 20.0  # Above standby (under 10 W here) and below real activity.
    max_gap_min: int = 10  # Off-gaps cluster at 1 to 5 minutes. Longer tolerances merge separate washes.
    min_duration_min: int = 20  # Shorter bursts hold about 1% of energy and aren't full programmes.
    min_peak_w: float = 200.0  # Keeps unheated rinse or cold cycles and rejects flat held values.
    heat_w: float = 1000.0  # 1-minute mean at or above this is treated as heating.
    min_heat_minutes: int = 2  # Heating shorter than this does not count as a heated cycle.


DEFAULT_PARAMS = CycleParams()
CYCLE_COLUMNS = ["start_pos", "end_pos", "duration_min", "on_minutes", "peak_w", "energy_kwh", "heat_minutes",
                 "heat_energy_kwh", "heated", "complete"]


def on_runs(power: np.ndarray, on_w: float, max_gap_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Start and end positions of on-runs on a complete minute grid, with short gaps bridged."""
    on = np.flatnonzero(np.nan_to_num(power, nan=-1.0) > on_w)
    if len(on) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    nan_before = np.cumsum(np.isnan(power))
    gap_length = np.diff(on) - 1
    has_missing = nan_before[on[1:]] - nan_before[on[:-1]] > 0
    breaks = np.flatnonzero((gap_length > max_gap_min) | has_missing)
    return on[np.r_[0, breaks + 1]], on[np.r_[breaks, len(on) - 1]]


def detect_cycles(power: np.ndarray, params: CycleParams = DEFAULT_PARAMS) -> pd.DataFrame:
    """Return one row per detected cycle. `power` is in watts, on a complete 1-minute grid."""
    power = np.asarray(power, dtype="float64")
    starts, ends = on_runs(power, params.on_w, params.max_gap_min)
    rows = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        duration = end - start + 1
        if duration < params.min_duration_min:
            continue
        segment = power[start:end + 1]
        peak = float(np.nanmax(segment))
        if peak < params.min_peak_w:
            continue
        hot = segment >= params.heat_w
        before = power[max(0, start - params.max_gap_min):start]
        after = power[end + 1:end + 1 + params.max_gap_min]
        heat_minutes = int(hot.sum())
        rows.append({
            "start_pos": start,
            "end_pos": end,
            "duration_min": duration,
            "on_minutes": int((segment > params.on_w).sum()),
            "peak_w": peak,
            "energy_kwh": float(np.nansum(segment) / 60000),
            "heat_minutes": heat_minutes,
            "heat_energy_kwh": float(segment[hot].sum() / 60000),
            "heated": heat_minutes >= params.min_heat_minutes,
            "complete": not (np.isnan(before).any() or np.isnan(after).any()),
        })
    return pd.DataFrame(rows, columns=CYCLE_COLUMNS)


def params_dict(params: CycleParams = DEFAULT_PARAMS) -> dict:
    return asdict(params)
