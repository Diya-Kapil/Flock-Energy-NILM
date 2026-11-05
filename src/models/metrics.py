"""Evaluation metrics. Power in watts and energy in kWh."""
from __future__ import annotations

import numpy as np
import pandas as pd

ON_W = 20.0  # Same threshold as cycle detection, so "operating" has one meaning throughout.
THRESHOLD_GRID_W = (5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000)


def on_off(actual_on: np.ndarray, predicted_on: np.ndarray) -> dict:
    tp = int((actual_on & predicted_on).sum())
    fp = int((~actual_on & predicted_on).sum())
    fn = int((actual_on & ~predicted_on).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(actual: np.ndarray, predicted: np.ndarray, threshold_w: float, step_minutes: int = 1, on_w: float = ON_W) -> dict:
    """All power, on/off and energy metrics for one set of minutes."""
    y, p = np.asarray(actual, dtype="float64"), np.asarray(predicted, dtype="float64")
    error = p - y
    active = y > on_w
    ss_res = float((error**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) if len(y) else 0.0
    to_kwh = step_minutes / 60000
    actual_kwh, predicted_kwh = float(y.sum() * to_kwh), float(p.sum() * to_kwh)
    return {
        "minutes": len(y) * step_minutes,
        "active_minutes": int(active.sum()) * step_minutes,
        "mae_w": float(np.abs(error).mean()) if len(y) else np.nan,
        "rmse_w": float(np.sqrt((error**2).mean())) if len(y) else np.nan,
        "r2": 1 - ss_res / ss_tot if ss_tot else np.nan,
        "mae_active_w": float(np.abs(error[active]).mean()) if active.any() else np.nan,
        "mean_prediction_active_w": float(p[active].mean()) if active.any() else np.nan,
        "mean_prediction_off_w": float(p[~active].mean()) if (~active).any() else np.nan,
        "actual_kwh": actual_kwh,
        "predicted_kwh": predicted_kwh,
        "energy_error_pct": (predicted_kwh - actual_kwh) / actual_kwh * 100 if actual_kwh else np.nan,
        **on_off(active, p > threshold_w),
    }


def best_threshold(actual: np.ndarray, predicted: np.ndarray, on_w: float = ON_W, grid=THRESHOLD_GRID_W) -> tuple[float, float]:
    """On/off threshold on predictions with the highest F1. Call with training data only."""
    active = np.asarray(actual) > on_w
    scores = [(on_off(active, np.asarray(predicted) > t)["f1"], t) for t in grid]
    f1, threshold = max(scores, key=lambda item: (item[0], -item[1]))
    return float(threshold), float(f1)


def match_cycles(actual: pd.DataFrame, predicted: pd.DataFrame, min_overlap: float = 0.5) -> dict:
    """Match predicted to actual cycles, each used once, by overlap with the shorter of the two."""
    tp = 0
    matched_predictions: set[int] = set()
    matched_flags = []
    pred_starts = predicted["start_pos"].to_numpy() if len(predicted) else np.array([], dtype=int)
    pred_ends = predicted["end_pos"].to_numpy() if len(predicted) else np.array([], dtype=int)
    for _, cycle in actual.iterrows():
        start, end = int(cycle["start_pos"]), int(cycle["end_pos"])
        overlap = np.minimum(pred_ends, end) - np.maximum(pred_starts, start) + 1
        shorter = np.minimum(pred_ends - pred_starts + 1, end - start + 1)
        share = np.where(overlap > 0, overlap / shorter, 0.0) if len(pred_starts) else np.array([])
        order = np.argsort(-share) if len(share) else []
        hit = False
        for index in order:
            if share[index] < min_overlap:
                break
            if int(index) not in matched_predictions:
                matched_predictions.add(int(index))
                hit = True
                break
        tp += hit
        matched_flags.append(hit)
    fp = len(predicted) - len(matched_predictions)
    fn = len(actual) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "actual_cycles": len(actual),
        "predicted_cycles": len(predicted),
        "matched": tp,
        "cycle_precision": precision,
        "cycle_recall": recall,
        "cycle_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched_flags": matched_flags,
        "matched_prediction_positions": sorted(matched_predictions),
    }
