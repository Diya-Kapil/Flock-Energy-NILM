"""Size the energy-saving opportunity from measured cycles, and how hard it is to measure.

Savings are scenarios, not forecasts. They compare each household's cycles with that same
household's own lower-heat cycles, so no outside energy or tariff figure is assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_ALPHA_POWER_SQUARED = (1.96 + 0.8416) ** 2  # Two-sided 5% significance, 80% power.
PEAK_START, PEAK_END = 16, 19


def own_cycle_scenarios(cycles: pd.DataFrame, valid_days: float, quantile: float) -> dict:
    """Energy saved a year if heating above this house's own quantile were capped at it."""
    full = cycles[cycles["complete"] & cycles["heated"]]
    cap = float(full["heat_energy_kwh"].quantile(quantile))
    per_cycle = (full["heat_energy_kwh"] - cap).clip(lower=0)
    scale = 365 / valid_days * len(cycles[cycles["heated"]]) / max(len(full), 1)
    return {"cap_heat_kwh_per_cycle": round(cap, 3), "cycles_reduced_pct": round(float((per_cycle > 0).mean() * 100), 1),
            "saving_kwh_per_year": round(float(per_cycle.sum() * scale), 1)}


def peak_figures(table: pd.DataFrame) -> dict:
    """Machine use in the 16:00 to 19:00 UK local window, on days with a usable clock."""
    usable = table[table["valid"] & table["uk_local_minute"].notna()]
    hours = usable["uk_local_minute"].dt.hour
    peak = (hours >= PEAK_START) & (hours < PEAK_END)
    power, aggregate = usable["washing_machine_w"].astype("float64"), usable["aggregate_w"].astype("float64")
    days = len(usable) / 1440
    return {
        "machine_energy_in_peak_pct": round(float(power[peak].sum() / power.sum() * 100), 1),
        "machine_kwh_in_peak_per_year": round(float(power[peak].sum() / 60000 / days * 365), 1),
        "mean_machine_w_in_peak": round(float(power[peak].mean()), 1),
        "machine_share_of_peak_aggregate_pct": round(float(power[peak].sum() / aggregate[peak].sum() * 100), 2),
    }


def opportunity_table(cycle_summary: pd.DataFrame, cycles: pd.DataFrame, tables: dict[int, pd.DataFrame], power_states: pd.DataFrame) -> pd.DataFrame:
    """One row per household: annual energy, heating share, scenarios and peak figures."""
    rows = []
    states = power_states.set_index("house")
    for _, summary in cycle_summary.iterrows():
        house = int(summary["house"])
        own = cycles[cycles["house"] == house]
        median = own_cycle_scenarios(own, summary["valid_days"], 0.5)
        quartile = own_cycle_scenarios(own, summary["valid_days"], 0.25)
        machine = float(summary["machine_kwh_per_year"])
        household = float(states.loc[house, "aggregate_kwh_per_day"]) * 365
        rows.append({
            "house": house, "occupancy": int(summary["occupancy"]), "household_kwh_per_year": round(household, 0), "machine_kwh_per_year": machine,
            "machine_share_of_household_pct": float(states.loc[house, "machine_share_of_aggregate_energy_pct"]),
            "heat_share_of_cycle_energy_pct": float(summary["heat_share_of_cycle_energy_pct"]),
            "median_cycle_kwh": float(summary["median_energy_kwh"]), "p90_cycle_kwh": float(summary["p90_energy_kwh"]),
            "cap_at_own_median_kwh_per_cycle": median["cap_heat_kwh_per_cycle"], "saving_cap_at_median_kwh_per_year": median["saving_kwh_per_year"],
            "saving_cap_at_median_pct_of_machine": round(median["saving_kwh_per_year"] / machine * 100, 1),
            "saving_cap_at_median_pct_of_household": round(median["saving_kwh_per_year"] / household * 100, 2),
            "saving_cap_at_quartile_kwh_per_year": quartile["saving_kwh_per_year"],
            "saving_cap_at_quartile_pct_of_machine": round(quartile["saving_kwh_per_year"] / machine * 100, 1),
            **peak_figures(tables[house]),
        })
    return pd.DataFrame(rows)


def sample_size_per_arm(coefficient_of_variation: float, effect_fraction: float, variance_kept: float = 1.0) -> int:
    """Households per arm to detect a fractional change, by the two-sample normal formula.

    `variance_kept` is the share of variance left after adjusting for pre-trial use.
    """
    if effect_fraction <= 0:
        raise ValueError("Effect must be positive.")
    return int(np.ceil(2 * Z_ALPHA_POWER_SQUARED * variance_kept * (coefficient_of_variation / effect_fraction) ** 2))


def measurement_power(opportunity: pd.DataFrame, pre_period_variance_kept: float = 0.3) -> pd.DataFrame:
    """Compare measuring the same saving at the washing machine and at the whole-house meter."""
    machine, household = opportunity["machine_kwh_per_year"], opportunity["household_kwh_per_year"]
    saving = float(opportunity["saving_cap_at_median_kwh_per_year"].mean())
    rows = []
    for name, values in (("Washing machine sub-meter", machine), ("Whole-house smart meter", household)):
        mean, cv = float(values.mean()), float(values.std(ddof=1) / values.mean())
        effect = saving / mean
        rows.append({"measured_at": name, "mean_kwh_per_year": round(mean, 0), "between_home_cv": round(cv, 2), "saving_kwh_per_year": round(saving, 1),
                     "saving_as_pct_of_mean": round(effect * 100, 2), "households_per_arm_unadjusted": sample_size_per_arm(cv, effect),
                     "households_per_arm_with_pre_period_adjustment": sample_size_per_arm(cv, effect, pre_period_variance_kept)})
    return pd.DataFrame(rows)
