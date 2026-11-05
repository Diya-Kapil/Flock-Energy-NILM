"""Features built from aggregate household power, and nothing else.

`make_features` accepts one array of aggregate watts, so appliance readings, Issues and
household identity cannot reach the model. Every primary feature looks backwards only. Future
readings are used solely when `retrospective=True`, which marks a separately labelled
experiment for after-the-fact analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_MINUTES = (1, 2, 5, 15, 30, 60)
WINDOW_MINUTES = (5, 15, 30, 60, 120)
FUTURE_WINDOW_MINUTES = (15, 30, 60, 120)
HEATER_RISE_W = 1500.0  # Washing machine heaters here draw about 2 kW.
ACTIVITY_STEP_W = 100.0
LARGE_STEP_W = 1000.0
SINCE_RISE_CAP_MINUTES = 240


def _steps(minutes: int, step_minutes: int) -> int:
    return max(1, minutes // step_minutes)


def make_features(aggregate: np.ndarray, step_minutes: int = 1, retrospective: bool = False) -> pd.DataFrame:
    """Build features from aggregate watts on a complete, regular time grid.

    Missing readings are NaN and stay NaN. Rolling windows need half their readings. Scale
    comes only from the same house's own past readings, so nothing is fitted on other data.
    """
    a = pd.Series(np.asarray(aggregate, dtype="float64"))
    change = a.diff()
    out: dict[str, pd.Series] = {"aggregate": a}

    lags = sorted({_steps(m, step_minutes) for m in LAG_MINUTES if m >= step_minutes} or {1})
    for k in lags:
        label = k * step_minutes
        out[f"lag_{label}m"] = a.shift(k)
        out[f"change_{label}m"] = a - a.shift(k)

    windows = sorted({_steps(m, step_minutes) for m in WINDOW_MINUTES if m >= 2 * step_minutes})
    for w in windows:
        label = w * step_minutes
        rolling = a.rolling(w, min_periods=max(1, w // 2))
        out[f"mean_{label}m"] = rolling.mean()
        out[f"std_{label}m"] = rolling.std()
        out[f"min_{label}m"] = rolling.min()
        out[f"max_{label}m"] = rolling.max()
        out[f"above_min_{label}m"] = a - out[f"min_{label}m"]

    long_window = windows[-1] if windows else 1
    baseline = a.rolling(long_window, min_periods=1).min()
    span_60 = _steps(60, step_minutes)
    span_30 = _steps(30, step_minutes)
    out["large_rises_60m"] = (change >= LARGE_STEP_W).astype("float64").rolling(span_60, min_periods=1).sum()
    out["activity_steps_30m"] = (change.abs() >= ACTIVITY_STEP_W).astype("float64").rolling(span_30, min_periods=1).sum()
    out["high_load_minutes_60m"] = ((a - baseline) >= HEATER_RISE_W).astype("float64").rolling(span_60, min_periods=1).sum() * step_minutes

    positions = np.arange(len(a), dtype="float64")
    last_rise = pd.Series(np.where(change.to_numpy() >= HEATER_RISE_W, positions, np.nan)).ffill().to_numpy()
    since = (positions - last_rise) * step_minutes
    out["minutes_since_heater_rise"] = pd.Series(np.nan_to_num(np.minimum(since, SINCE_RISE_CAP_MINUTES), nan=SINCE_RISE_CAP_MINUTES))

    if retrospective:
        reverse = a[::-1].reset_index(drop=True)
        for k in lags:
            out[f"future_change_{k * step_minutes}m"] = a.shift(-k) - a
        for w in sorted({_steps(m, step_minutes) for m in FUTURE_WINDOW_MINUTES if m >= 2 * step_minutes}):
            label = w * step_minutes
            rolling = reverse.rolling(w, min_periods=max(1, w // 2))
            for name, values in (("mean", rolling.mean()), ("std", rolling.std()), ("max", rolling.max()), ("min", rolling.min())):
                out[f"future_{name}_{label}m"] = pd.Series(values.to_numpy()[::-1])
    return pd.DataFrame(out).astype("float32")


def feature_names(step_minutes: int = 1, retrospective: bool = False) -> list[str]:
    return list(make_features(np.arange(5, dtype="float64"), step_minutes, retrospective).columns)
