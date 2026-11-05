"""Phase 2 dataset inspection for REFIT. Read-only: nothing here cleans or models data.

Regenerates outputs/reports/dataset_inventory.csv and household_screening.csv from
the downloaded archives. CSVs are decompressed into memory one at a time, so the
archives stay untouched and no extracted file is written to disk.

Run from the project root:
    python -m src.cleaning.inspect_dataset
    python -m src.cleaning.inspect_dataset --plot-dir /some/folder   # adds review plots
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import py7zr

from src.io_utils import stream_members

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / "data" / "raw" / "archives"
DOCS = ROOT / "data" / "raw" / "docs"
REPORTS = ROOT / "outputs" / "reports"
RAW_ARCHIVE = ARCHIVES / "REFIT_RAW_081116.7z"
CLEAN_ARCHIVE = ARCHIVES / "CLEAN_REFIT_081116.7z"
RAW_README = DOCS / "RAW_READ_ME_081116.txt"
CLEAN_README = DOCS / "CLEAN_READ_ME_081116.txt"
METADATA = DOCS / "MetaData_Tables.xlsx"

HOUSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]
APPLIANCES = [f"Appliance{i}" for i in range(1, 10)]

# Screening event definition. This is a screening aid, not the final cycle detector.
ON_W = 20
MERGE_GAP_S = 900
HEAT_W = 1000
HEAT_BLOCK_GAP_S = 300
HEAT_BLOCK_MIN_S = 120
EVENT_MIN_MINUTES = 20
EVENT_MAX_MINUTES = 300
EVENT_MIN_PEAK_W = 500
DT_CAP_S = 60  # Energy is not integrated across row gaps longer than this.

# Gate thresholds. These were set after viewing the distributions across all houses, so
# they describe the data and are not an independent validation of the labels.
SIGNATURE_PEAK_W = (1800, 3000)
SIGNATURE_DURATION_MIN = (20, 240)
SIGNATURE_MAX_HEAT_BLOCKS = 1
SIGNATURE_MIN_LOW_POWER_SHARE = 0.5
MIN_EVENTS = 100
MIN_EVENTS_PER_WEEK = 1.0
MIN_DATA_DAYS = 300
MIN_GOOD_MONTHS = 12
GOOD_MONTH_DAYS = 10
MIN_ACTIVE_MONTH_SHARE = 0.8
EDGE_GAIN_RANGE = (0.7, 1.3)
MAX_IAM_ABOVE_AGG_HEAT_PCT = 20.0
MAX_QUARTER_PEAK_RATIO = 1.15
MAX_QUARTER_DURATION_RATIO = 1.75
QUARTER_MIN_DAYS = 30
QUARTER_MIN_EVENTS = 10

# Analyst recommendation among houses that pass every gate, pending user approval.
RECOMMENDED_PRIMARY = [1, 2, 8, 10, 15, 16]

# Manual findings from reviewing plots of seeded random events (--plot-dir reproduces them).
VISUAL_NOTES = {
    (2, 2): "Consistent with a washing machine; some straight segments across row gaps.",
    (4, 4): "Consistent with a washing machine; heating starts at once, unlike most.",
    (4, 5): "Consistent with a washing machine; heating starts at once, unlike most.",
    (5, 3): "Consistent with a washing machine; long flat low-power stretches suggest held values.",
    (13, 3): "Consistent with a washing machine; some flat stretches suggest held values.",
    (16, 5): "Consistent with a washing machine; sparse low-power activity.",
    (19, 2): "Consistent with a washing machine; very short heating, about 2 minutes.",
    (20, 4): "Consistent with a washing machine; sparse low-power activity.",
}
VISUAL_DEFAULT = "Consistent with a washing machine."
REVIEW_OTHER = [(2, 3), (7, 6), (10, 6), (3, 4), (7, 4), (20, 3), (1, 4), (8, 3),
                (9, 2), (18, 4), (5, 7), (13, 2), (13, 5), (16, 9), (1, 6), (5, 2)]

# Published md5s from the Zenodo mirror (record 5063428), for an independent integrity check.
ZENODO_MD5 = {
    "CLEAN_House1.csv": "f5336d80c700f866af0d997ea8d39146",
    "CLEAN_House2.csv": "13957fd139bf8f6ba8bceea36e85bb3d",
    "CLEAN_House3.csv": "ab6e34ea386eb4563bfcb9762619dc8f",
    "CLEAN_House4.csv": "efbfab2b04a61fbf7595422f15cacfb1",
    "CLEAN_House5.csv": "03938c844fdb09d787eb637bcf65f956",
    "CLEAN_House11.csv": "5b41bae97ab28871fcfe5167b41c0347",
}


# --------------------------------------------------------------------------- documents
def parse_readme(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    """Return {(house, channel): {label, make_model, note}} from an official readme."""
    out: dict[tuple[int, int], dict[str, str]] = {}
    house = None
    for line in path.read_text(encoding="latin-1").splitlines():
        text = line.strip()
        header = re.fullmatch(r"House (\d+)", text)
        if header:
            house = int(header[1])
            continue
        item = re.match(r"([1-9])\.(.*)", text)
        if house is None or item is None:
            continue
        body = item[2].strip().rstrip(",")
        note = re.search(r"\(((?:[Cc]hange|Starts|Stops|Nothing|Kettle changed)[^)]*)\)", body)
        body = body.replace(note[0], "").strip().rstrip(",") if note else body
        fields = [f.strip() for f in body.split(",")]
        out[(house, int(item[1]))] = {
            "label": fields[0],
            "make_model": " ".join(f for f in fields[1:] if f),
            "note": note[1] if note else "",
        }
    return out


def parse_metadata(path: Path) -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, str]]]:
    """Return the household table and {(house, channel): {label, make_model, note}}."""
    book = openpyxl.load_workbook(path, data_only=True, read_only=False)
    rows = list(book.worksheets[0].iter_rows(values_only=True))
    households = pd.DataFrame(rows[2:], columns=["house", "occupancy", "construction_year",
                                                 "appliances_owned", "building_type", "size"])
    households = households.dropna(subset=["house"]).astype({"house": int})
    appliances: dict[tuple[int, int], dict[str, str]] = {}
    for sheet in book.worksheets[1:]:
        house = int(sheet.title.split()[1])
        for row in sheet.iter_rows(values_only=True):
            if row[0] in (None, 0):
                continue
            cells = ["" if c is None else str(c).strip() for c in row] + ["", "", ""]
            note = cells[4] if cells[4].startswith("(") else ""
            appliances[(house, int(row[0]))] = {
                "label": cells[1],
                "make_model": " ".join(c for c in cells[2:4] if c),
                "note": note.strip("()"),
            }
    return households, appliances


# --------------------------------------------------------------------------- streaming
def house_of(name: str) -> int:
    return int(re.search(r"House_?(\d+)", name)[1])


# --------------------------------------------------------------------------- file facts
def file_facts(df: pd.DataFrame, data: bytes, listing: dict) -> dict:
    """Format and sampling facts for one CSV, measured from its contents."""
    u = df["Unix"].to_numpy()
    dt = np.diff(u)
    positive = dt[dt > 0]
    apps = df[APPLIANCES]
    aggregate = df["Aggregate"]
    long_gap = dt > 120
    observed_s = (u.max() - u.min()) - dt[long_gap].sum()
    unique_rows = df["Unix"].nunique()
    as_utc = df["Time"].astype("datetime64[s]").astype("int64").to_numpy()
    crc = zlib.crc32(data)
    return {
        "uncompressed_bytes": len(data),
        "crc32_matches_archive": crc == listing["crc32"],
        "md5": hashlib.md5(data).hexdigest(),
        "rows": len(df),
        "columns": "|".join(df.columns),
        "line_ending": "CRLF" if b"\r\n" in data[:4096] else "LF",
        "has_nan_token": b"NaN" in data,
        "first_time": str(df["Time"].iloc[0]),
        "last_time": str(df["Time"].iloc[-1]),
        "time_minus_unix_max_abs_s": int(np.abs(as_utc - u).max()),
        "non_monotonic_steps": int((dt < 0).sum()),
        "duplicate_timestamps": int(df["Unix"].duplicated().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "median_positive_gap_s": float(np.median(positive)),
        "p95_positive_gap_s": float(np.percentile(positive, 95)),
        "effective_seconds_per_unique_row": round(float(observed_s / max(unique_rows, 1)), 2),
        "gaps_gt_2min": int(long_gap.sum()),
        "days_in_gaps_gt_2min": round(float(dt[long_gap].sum() / 86400), 1),
        "gaps_gt_1h": int((dt > 3600).sum()),
        "gaps_gt_1d": int((dt > 86400).sum()),
        "days_in_gaps_gt_1d": round(float(dt[dt > 86400].sum() / 86400), 1),
        "aggregate_nan_pct": round(float(aggregate.isna().mean() * 100), 2),
        "appliance_nan_pct_mean": round(float(apps.isna().mean().mean() * 100), 2),
        "aggregate_zero_pct": round(float((aggregate == 0).mean() * 100), 4),
        "aggregate_max_w": float(aggregate.max()),
        "appliance_max_w": float(apps.max().max()),
        "rows_any_appliance_gt_4000w": int((apps > 4000).any(axis=1).sum()),
        "rows_aggregate_gt_23000w": int((aggregate > 23000).sum()),
    }


def minute_mean(u: np.ndarray, values: np.ndarray) -> pd.Series:
    keep = ~np.isnan(values)
    return pd.Series(values[keep]).groupby(u[keep] // 60).mean()


# --------------------------------------------------------------------------- channel checks
def find_events(u: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    on = np.flatnonzero(p > ON_W)
    if len(on) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(u[on]) > MERGE_GAP_S)
    return on[np.r_[0, breaks + 1]], on[np.r_[breaks, len(on) - 1]]


def describe_event(u: np.ndarray, p: np.ndarray, start: int, end: int) -> dict:
    x, t = p[start:end + 1], u[start:end + 1]
    dt = np.minimum(np.diff(t, append=t[-1]), DT_CAP_S)
    on, hot = x > ON_W, x >= HEAT_W
    on_s = max(dt[on].sum(), 1)
    blocks = 0
    hot_rows = np.flatnonzero(hot)
    if len(hot_rows):
        cut = np.flatnonzero(np.diff(t[hot_rows]) > HEAT_BLOCK_GAP_S)
        first = hot_rows[np.r_[0, cut + 1]]
        last = hot_rows[np.r_[cut, len(hot_rows) - 1]]
        blocks = int(((t[last] - t[first]) >= HEAT_BLOCK_MIN_S).sum())
    minutes = (t[-1] - t[0]) / 60
    return {
        "start": int(t[0]),
        "duration_min": minutes,
        "kwh": float((x * dt).sum() / 3.6e6),
        "peak_w": float(np.percentile(x, 99.5)),
        "heat_min": float(dt[hot].sum() / 60),
        "heat_blocks": blocks,
        "low_power_share": float(dt[on & (x < 500)].sum() / on_s),
        "steps_per_hour": float((np.abs(np.diff(x)) >= 100).sum() / max(minutes / 60, 0.01)),
    }


def long_events(u: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    starts, ends = find_events(u, p)
    minutes = (u[ends] - u[starts]) / 60
    picked = [describe_event(u, p, a, b) for a, b, m in zip(starts, ends, minutes)
              if EVENT_MIN_MINUTES <= m <= EVENT_MAX_MINUTES and p[a:b + 1].max() >= EVENT_MIN_PEAK_W]
    return pd.DataFrame(picked)


def matches_signature(events: pd.DataFrame) -> bool:
    if len(events) < 30:
        return False
    return bool(SIGNATURE_PEAK_W[0] <= events.peak_w.median() <= SIGNATURE_PEAK_W[1]
                and SIGNATURE_DURATION_MIN[0] <= events.duration_min.median() <= SIGNATURE_DURATION_MIN[1]
                and events.heat_blocks.median() <= SIGNATURE_MAX_HEAT_BLOCKS
                and events.low_power_share.median() >= SIGNATURE_MIN_LOW_POWER_SHARE)


def best_lag(p: np.ndarray, a: np.ndarray, mask: np.ndarray, lags) -> int | None:
    """Lag in minutes at which aggregate best follows the channel on a shared 1-minute grid."""
    if mask.sum() < 3000:
        return None
    best, best_r = None, -2.0
    for lag in lags:
        y = np.roll(a, -lag)
        ok = mask & ~np.isnan(p) & ~np.isnan(y)
        if ok.sum() < 3000 or np.std(p[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        r = np.corrcoef(p[ok], y[ok])[0, 1]
        if r > best_r:
            best, best_r = lag, r
    return best


def stamp(day: str) -> int:
    return int(pd.Timestamp(day).value // 10**9)


def washing_machine_row(df: pd.DataFrame, channel: int, events: pd.DataFrame, dt: np.ndarray,
                        good_months: set[str], data_days: float) -> dict:
    """All data checks for one documented washing machine channel."""
    col = f"Appliance{channel}"
    u = df["Unix"].to_numpy()
    p = df[col].to_numpy().astype(float)
    a = df["Aggregate"].to_numpy().astype(float)
    total = df[APPLIANCES].sum(axis=1).to_numpy().astype(float)
    issues = df["Issues"].to_numpy()
    row: dict = {}

    row["wm_candidate_events"] = len(events)
    if len(events):
        starts = np.sort(events.start.to_numpy())
        months = pd.to_datetime(events.start, unit="s").dt.to_period("M").astype(str)
        ends = events.sort_values("start").pipe(lambda e: e.start + e.duration_min * 60).to_numpy()
        cumulative_days = np.cumsum(dt) / 86400
        at = np.searchsorted(u, starts)
        silence = np.diff(np.r_[0, cumulative_days[at], cumulative_days[-1]])
        row.update({
            "wm_events_per_week_of_data": round(len(events) / (data_days / 7), 2),
            "wm_median_duration_min": round(events.duration_min.median(), 1),
            "wm_p10_duration_min": round(events.duration_min.quantile(0.1), 1),
            "wm_p90_duration_min": round(events.duration_min.quantile(0.9), 1),
            "wm_median_kwh": round(events.kwh.median(), 3),
            "wm_median_peak_w": round(events.peak_w.median(), 0),
            "wm_median_heat_min": round(events.heat_min.median(), 1),
            "wm_median_heat_blocks": float(events.heat_blocks.median()),
            "wm_median_low_power_share": round(events.low_power_share.median(), 2),
            "wm_median_steps_per_hour": round(events.steps_per_hour.median(), 1),
            "wm_months_active": len(set(months) & good_months),
            "wm_longest_silence_data_days": round(float(silence.max()), 1),
            "wm_events_within_1h_of_previous_pct": round(float(np.mean((starts[1:] - ends[:-1]) < 3600) * 100), 1),
        })
        quarters = pd.to_datetime(events.start, unit="s").dt.to_period("Q").astype(str)
        table = events.groupby(quarters.to_numpy()).agg(n=("kwh", "size"), peak=("peak_w", "median"),
                                                        duration=("duration_min", "median"))
        table["days"] = pd.Series(dt).groupby(pd.to_datetime(u, unit="s").to_period("Q").astype(str)).sum() / 86400
        stable = table[(table.days >= QUARTER_MIN_DAYS) & (table.n >= QUARTER_MIN_EVENTS)]
        if len(stable):
            row["wm_quarter_peak_ratio"] = round(stable.peak.max() / stable.peak.min(), 2)
            row["wm_quarter_duration_ratio"] = round(stable.duration.max() / stable.duration.min(), 2)

    edges = np.flatnonzero(np.diff(p) >= HEAT_W)
    edges = edges[(edges >= 3) & (edges < len(p) - 5)]
    if len(edges):
        gain = (a[edges + 4] - a[edges - 3]) / (p[edges + 1] - p[edges])
        row["edge_count"] = len(edges)
        row["edge_gain_median"] = round(float(np.median(gain)), 2)
        row["edge_gain_p25"] = round(float(np.percentile(gain, 25)), 2)
        row["edge_gain_p75"] = round(float(np.percentile(gain, 75)), 2)

    hot, on = p >= HEAT_W, p > ON_W
    if hot.any():
        row["aggregate_minus_wm_during_heat_median_w"] = round(float(np.median((a - p)[hot])), 0)
        row["aggregate_below_wm_during_heat_pct"] = round(float(np.mean(a[hot] < p[hot]) * 100), 1)
    row["issues_pct_wm_on_rows"] = round(float(issues[on].mean() * 100), 2) if on.any() else None
    row["issues_pct_wm_ge_500w_rows"] = round(float(issues[p >= 500].mean() * 100), 2) if (p >= 500).any() else None
    row["issues_pct_wm_heat_rows"] = round(float(issues[hot].mean() * 100), 2) if hot.any() else None

    minute = pd.DataFrame({"m": u // 60, "p": p, "a": a, "s": total}).groupby("m").mean()
    heat_minutes = minute.p >= HEAT_W
    row["iam_sum_above_aggregate_heat_minutes_pct"] = (
        round(float((minute.s[heat_minutes] > minute.a[heat_minutes]).mean() * 100), 1) if heat_minutes.any() else None)
    grid = np.arange(minute.index[0], minute.index[-1] + 1)
    gp, ga = minute.p.reindex(grid).to_numpy(), minute.a.reindex(grid).to_numpy()
    seconds = grid * 60
    near = list(range(-10, 11)) + [-60, 60]
    row["aggregate_lag_overall_min"] = best_lag(gp, ga, np.ones(len(grid), bool), range(-90, 91))
    for name, lo, hi in (("summer_2014", "2014-04-05", "2014-08-25"), ("winter_2015", "2014-11-01", "2015-03-25"),
                         ("summer_2015", "2015-04-05", "2015-07-15")):
        row[f"aggregate_lag_{name}_min"] = best_lag(gp, ga, (seconds >= stamp(lo)) & (seconds < stamp(hi)), near)
    return row


class Inspector:
    """Collects facts across both archive passes."""

    def __init__(self, plot_dir: Path | None):
        self.raw_readme = parse_readme(RAW_README)
        self.clean_readme = parse_readme(CLEAN_README)
        self.households, self.excel = parse_metadata(METADATA)
        self.facts: dict[str, dict] = {}
        self.listing: dict[str, dict] = {}
        self.house_rows: dict[int, dict] = {}
        self.channel_rows: list[dict] = []
        self.raw_minutes: dict[int, list[pd.Series]] = defaultdict(list)
        self.clean_minutes: dict[int, pd.Series] = {}
        self.plot_dir = plot_dir
        self.traces: dict[tuple[int, int], dict] = {}

    def labels(self, house: int) -> dict[int, str]:
        return {c: self.excel[(house, c)]["label"] for c in range(1, 10)}

    def on_raw(self, name: str, data: bytes) -> None:
        df = pd.read_csv(io.BytesIO(data), engine="pyarrow")
        self.facts[name] = file_facts(df, data, self.listing[name])
        house = house_of(name)
        self.raw_minutes[house].append(minute_mean(df["Unix"].to_numpy(), df["Aggregate"].to_numpy().astype(float)))
        print(f"raw   {name}: {len(df):,} rows", flush=True)

    def on_clean(self, name: str, data: bytes) -> None:
        df = pd.read_csv(io.BytesIO(data), engine="pyarrow")
        self.facts[name] = file_facts(df, data, self.listing[name])
        self.facts[name]["null_cells"] = int(df.isna().sum().sum())
        house = house_of(name)
        u = df["Unix"].to_numpy()
        a = df["Aggregate"].to_numpy().astype(float)
        self.clean_minutes[house] = minute_mean(u, a)

        dt = np.minimum(np.diff(u, append=u[-1]), DT_CAP_S)
        data_days = float(dt.sum() / 86400)
        by_month = pd.Series(dt).groupby(pd.to_datetime(u, unit="s").to_period("M").astype(str)).sum() / 86400
        good_months = set(by_month[by_month >= GOOD_MONTH_DAYS].index)

        total = df[APPLIANCES].sum(axis=1).to_numpy().astype(float)
        issues = df["Issues"].to_numpy()
        zero = a == 0
        edges = np.flatnonzero(np.diff(np.r_[0, zero.astype(int), 0]))
        zero_runs_s = u[edges[1::2] - 1] - u[edges[::2]] if len(edges) else np.array([0])
        self.house_rows[house] = {
            "house": house,
            "clean_file": name,
            "clean_rows": len(df),
            "first_time": str(df["Time"].iloc[0]),
            "last_time": str(df["Time"].iloc[-1]),
            "span_days": round(float((u[-1] - u[0]) / 86400), 1),
            "data_days": round(data_days, 1),
            "data_coverage_pct": round(data_days / ((u[-1] - u[0]) / 86400) * 100, 1),
            "months_with_10d_data": len(good_months),
            "gaps_gt_2min": self.facts[name]["gaps_gt_2min"],
            "gaps_gt_1d": self.facts[name]["gaps_gt_1d"],
            "days_in_gaps_gt_1d": self.facts[name]["days_in_gaps_gt_1d"],
            "issues_pct_all_rows": round(float(issues.mean() * 100), 2),
            "issues_rule_match_pct": round(float((issues == (total > a)).mean() * 100), 4),
            "aggregate_zero_rows": int(zero.sum()),
            "aggregate_zero_longest_hours": round(float(zero_runs_s.max() / 3600), 1),
            "aggregate_max_w": float(a.max()),
            "aggregate_rows_gt_23kw": int((a > 23000).sum()),
        }

        labels = self.labels(house)
        events = {c: long_events(u, df[f"Appliance{c}"].to_numpy().astype(float)) for c in range(1, 10)}
        washing = [c for c, label in labels.items() if label.lower().startswith("washing machine")]
        self.house_rows[house]["other_channels_matching_wm_signature"] = "; ".join(
            f"ch{c} {labels[c]}" for c in range(1, 10) if c not in washing and matches_signature(events[c])) or "none"
        self.house_rows[house]["other_laundry_channels"] = "; ".join(
            f"ch{c} {labels[c]}: {len(events[c])} events, {len(events[c]) / (data_days / 7):.2f}/week"
            for c, label in labels.items() if re.match(r"(washer dryer|tumble dryer)", label.lower())) or "none"
        self.house_rows[house]["confounder_rate_per_week"] = round(sum(
            len(events[c]) / (data_days / 7) for c, label in labels.items()
            if re.match(r"(washer dryer|tumble dryer)", label.lower())), 2)
        for channel in washing:
            row = {"house": house, "wm_channel": channel,
                   "signature_match": matches_signature(events[channel])}
            row.update(washing_machine_row(df, channel, events[channel], dt, good_months, data_days))
            self.channel_rows.append(row)

        if self.plot_dir:
            for channel in washing + [c for h, c in REVIEW_OTHER if h == house]:
                self._keep_traces(house, channel, labels[channel], u, df[f"Appliance{channel}"].to_numpy().astype(float), events[channel])
        print(f"clean {name}: {len(df):,} rows", flush=True)

    def _keep_traces(self, house, channel, label, u, p, events) -> None:
        if not len(events):
            return
        chosen = events.sample(min(6, len(events)), random_state=42)
        traces = []
        for _, event in chosen.iterrows():
            i, j = np.searchsorted(u, event.start), np.searchsorted(u, event.start + event.duration_min * 60 + 60)
            traces.append(((u[i:j] - u[i]) / 60, p[i:j]))
        self.traces[(house, channel)] = {
            "label": label, "traces": traces, "n": len(events),
            "duration": events.duration_min.median(), "kwh": events.kwh.median()}

    def time_base(self, house: int) -> dict:
        """Compare the cleaned clock with the raw clock, and each with itself a year later."""
        raw = pd.concat(self.raw_minutes[house])
        raw = raw[~raw.index.duplicated()].sort_index()
        clean = self.clean_minutes[house]
        out: dict = {"house": house}
        for name, lo, hi in (("2014_05_01_to_06_25", "2014-05-01", "2014-06-25"),
                             ("2014_11_15_to_2015_03_15", "2014-11-15", "2015-03-15"),
                             ("2015_04_05_to_05_31", "2015-04-05", "2015-05-31")):
            grid = np.arange(stamp(lo) // 60, stamp(hi) // 60)
            x = clean.reindex(grid).to_numpy()
            best, best_r = None, -2.0
            for lag in (-60, 0, 60):
                y = raw.reindex(grid - lag).to_numpy()
                ok = ~np.isnan(x) & ~np.isnan(y)
                if ok.sum() < 1000 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                    continue
                r = np.corrcoef(x[ok], y[ok])[0, 1]
                if r > best_r:
                    best, best_r = lag, r
            out[f"clean_minus_raw_min_{name}"] = best

        def routine_shift(series: pd.Series) -> int | None:
            profiles = []
            for lo, hi in (("2014-04-05", "2014-05-31"), ("2015-04-05", "2015-05-31")):
                s = series[(series.index >= stamp(lo) // 60) & (series.index < stamp(hi) // 60)]
                if len(s) < 3000:
                    return None
                profiles.append(s.groupby((s.index % 1440) // 5).mean().reindex(range(288))
                                .interpolate(limit_direction="both").to_numpy())
            scores = [np.corrcoef(profiles[0], np.roll(profiles[1], -k))[0, 1] for k in range(-24, 25)]
            return (int(np.argmax(scores)) - 24) * 5

        out["routine_shift_2014_to_2015_clean_min"] = routine_shift(clean)
        out["routine_shift_2014_to_2015_raw_min"] = routine_shift(raw)
        return out


# --------------------------------------------------------------------------- report tables
def apply_gates(row: dict) -> dict:
    """Add gate results, status and written reasons to one screening row."""
    fail: list[str] = []
    if row["wm_channel"] is None or pd.isna(row["wm_channel"]):
        return {"gate_documents": False, "gate_signature": None, "gate_volume": None, "gate_coverage": None,
                "gate_stability": None, "gate_aggregate": None, "screening_status": "excluded",
                "status_reasons": "No washing machine is documented; three channels are labelled ???."}

    documents = bool(row["labels_agree_across_3_sources"]) and row["wm_channels_in_house"] == 1 and not row["doc_change_note"]
    if row["wm_channels_in_house"] > 1:
        fail.append("Two washing machines are metered, so one channel cannot represent the household target.")
    if row["doc_change_note"]:
        fail.append(f"Official note on this channel: {row['doc_change_note']}.")
    signature = bool(row["signature_match"])
    if not signature:
        fail.append("Event statistics fall outside the washing machine signature thresholds.")
    volume = row["wm_candidate_events"] >= MIN_EVENTS and row["wm_events_per_week_of_data"] >= MIN_EVENTS_PER_WEEK
    if not volume:
        fail.append(f"Only {int(row['wm_candidate_events'])} candidate events.")
    coverage = (row["data_days"] >= MIN_DATA_DAYS and row["months_with_10d_data"] >= MIN_GOOD_MONTHS
                and row["wm_months_active"] / row["months_with_10d_data"] >= MIN_ACTIVE_MONTH_SHARE)
    if not coverage:
        fail.append("Insufficient data coverage or too few active months.")
    excluded = bool(fail)

    stability = (row.get("wm_quarter_peak_ratio", 9) <= MAX_QUARTER_PEAK_RATIO
                 and row.get("wm_quarter_duration_ratio", 9) <= MAX_QUARTER_DURATION_RATIO)
    above_aggregate = row.get("iam_sum_above_aggregate_heat_minutes_pct", 100)
    aggregate = (EDGE_GAIN_RANGE[0] <= row.get("edge_gain_median", -9) <= EDGE_GAIN_RANGE[1]
                 and above_aggregate <= MAX_IAM_ABOVE_AGG_HEAT_PCT)
    limits: list[str] = []
    if not stability:
        limits.append(f"Median event duration changes {row.get('wm_quarter_duration_ratio')}x between quarters.")
    if not aggregate:
        limits.append(
            f"Aggregate tracks the channel poorly: edge gain {row.get('edge_gain_median')}, sensor sum above "
            f"aggregate in {row.get('iam_sum_above_aggregate_heat_minutes_pct')}% of heating minutes.")

    if excluded:
        status, reasons = "excluded", fail + limits
    elif limits:
        status, reasons = "eda_only", limits
    elif row["house"] in RECOMMENDED_PRIMARY:
        status, reasons = "recommended_primary", ["Passes every gate."]
    else:
        status, reasons = "reserve", ["Passes every gate; not needed for the recommended set."]
    return {"gate_documents": documents, "gate_signature": signature, "gate_volume": volume,
            "gate_coverage": coverage, "gate_stability": stability, "gate_aggregate": aggregate,
            "screening_status": status, "status_reasons": " ".join(reasons)}


def uncertainties(row: dict) -> str:
    notes: list[str] = []
    if row["wm_channel"] is None:
        return "The three channels labelled ??? could not be identified from the documents."
    if not row.get("doc_make_model") or "Unknown" in str(row.get("doc_make_model")):
        notes.append("Make or model is not documented.")
    if row.get("other_laundry_channels") not in (None, "none"):
        notes.append("Other laundry load is present in the aggregate.")
    if (row.get("issues_pct_wm_ge_500w_rows") or 0) >= 10:
        notes.append(f"Issues flags {row['issues_pct_wm_ge_500w_rows']}% of rows with the machine at 500 W or more.")
    if row.get("aggregate_rows_gt_23kw", 0) > 0:
        notes.append(f"{row['aggregate_rows_gt_23kw']} aggregate rows exceed 23 kW.")
    if row.get("aggregate_zero_longest_hours", 0) >= 1:
        notes.append(f"Aggregate stays at zero for up to {row['aggregate_zero_longest_hours']} hours.")
    if (row.get("wm_events_within_1h_of_previous_pct") or 0) >= 15:
        notes.append("Many events start soon after the previous one, so some may be split or back-to-back cycles.")
    gain = row.get("edge_gain_median")
    if gain is not None and gain == gain and not 0.9 <= gain <= 1.1:
        notes.append(f"Aggregate rises {gain} W per 1 W of channel at heating edges, so the two sensors disagree on scale.")
    return " ".join(notes)


def build_screening(ins: Inspector) -> pd.DataFrame:
    channels = pd.DataFrame(ins.channel_rows)
    times = pd.DataFrame([ins.time_base(h) for h in HOUSES])
    houses = pd.DataFrame(ins.house_rows.values()).merge(ins.households, on="house").merge(times, on="house")
    table = houses.merge(channels, on="house", how="left").sort_values(["house", "wm_channel"])
    records = []
    raw_files = {house_of(n): [] for n in ins.facts if n.startswith("RAW")}
    for name in sorted(ins.facts):
        if name.startswith("RAW"):
            raw_files[house_of(name)].append(name)
    for row in table.to_dict("records"):
        house = row["house"]
        channel = None if pd.isna(row["wm_channel"]) else int(row["wm_channel"])
        row["wm_channel"] = channel
        washing = [c for c, label in ins.labels(house).items() if label.lower().startswith("washing machine")]
        row["wm_channels_in_house"] = len(washing)
        row["raw_files"] = "; ".join(raw_files[house])
        if channel is not None:
            key = (house, channel)
            sources = (ins.raw_readme[key], ins.clean_readme[key], ins.excel[key])
            row["doc_raw_readme_label"], row["doc_clean_readme_label"], row["doc_excel_label"] = [s["label"] for s in sources]
            row["labels_agree_across_3_sources"] = len({s["label"] for s in sources}) == 1
            row["doc_make_model"] = ins.clean_readme[key]["make_model"]
            row["doc_change_note"] = ins.clean_readme[key]["note"]
            row["visual_review"] = VISUAL_NOTES.get(key, VISUAL_DEFAULT)
        else:
            row["doc_change_note"] = ""
            row["labels_agree_across_3_sources"] = True
            row["visual_review"] = "Not applicable: no channel documented as a washing machine."
        row.update(apply_gates(row))
        row["uncertainties"] = uncertainties(row)
        records.append(row)
    result = pd.DataFrame(records)
    whole_numbers = ["wm_channel", "wm_candidate_events", "wm_months_active", "edge_count"] + [
        c for c in result.columns if "_lag_" in c or c.startswith(("clean_minus_raw", "routine_shift"))]
    for column in whole_numbers:
        result[column] = pd.to_numeric(result[column]).astype("Int64")
    first = ["house", "wm_channel", "screening_status", "status_reasons", "uncertainties", "visual_review",
             "gate_documents", "gate_signature", "gate_volume", "gate_coverage", "gate_stability", "gate_aggregate",
             "occupancy", "construction_year", "appliances_owned", "building_type", "size",
             "doc_raw_readme_label", "doc_clean_readme_label", "doc_excel_label", "labels_agree_across_3_sources",
             "doc_make_model", "doc_change_note", "wm_channels_in_house", "other_laundry_channels",
             "confounder_rate_per_week", "other_channels_matching_wm_signature", "signature_match"]
    return result[first + [c for c in result.columns if c not in first]]


def build_inventory(ins: Inspector, screening: pd.DataFrame) -> pd.DataFrame:
    status = screening.groupby("house").screening_status.first().to_dict()
    roles = {
        RAW_ARCHIVE.name: ("archive", "raw", "All raw CSVs; only House 1 is needed."),
        CLEAN_ARCHIVE.name: ("archive", "cleaned", "All official cleaned CSVs."),
        RAW_README.name: ("document", "raw", "Raw format, appliance lists."),
        CLEAN_README.name: ("document", "cleaned", "Issues definition, cleaning summary, change notes."),
        METADATA.name: ("document", "metadata", "Household table and appliance lists."),
    }
    rows = []
    for path in (RAW_ARCHIVE, CLEAN_ARCHIVE, RAW_README, CLEAN_README, METADATA):
        kind, dataset, purpose = roles[path.name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"file_name": path.name, "record_type": kind, "dataset": dataset,
                     "location": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
                     "sha256": digest, "project_use": purpose})
    for name, facts in sorted(ins.facts.items(), key=lambda kv: (kv[0].split("House")[0], house_of(kv[0]), kv[0])):
        raw = name.startswith("RAW")
        house = house_of(name)
        part = re.search(r"Part(\d)", name)
        archive = RAW_ARCHIVE if raw else CLEAN_ARCHIVE
        if raw:
            use = "Raw House 1 cleaning" if house == 1 else "Not used; clock comparison only"
        else:
            use = {"recommended_primary": "EDA and modelling (recommended)", "reserve": "Reserve for EDA and modelling",
                   "eda_only": "EDA only", "excluded": "Excluded"}[status[house]]
        zenodo = ZENODO_MD5.get(name)
        rows.append({"file_name": name, "record_type": "archive_member", "dataset": "raw" if raw else "cleaned",
                     "location": f"{archive.relative_to(ROOT)}!{name} (not extracted)", "house": house,
                     "part": int(part[1]) if part else None, "crc32": ins.listing[name]["crc32"],
                     "matches_zenodo_md5": (facts["md5"] == zenodo) if zenodo else None, "project_use": use, **facts})
    table = pd.DataFrame(rows)
    table["house"] = table["house"].astype("Int64")
    table["part"] = table["part"].astype("Int64")
    first = ["file_name", "record_type", "dataset", "location", "house", "part", "project_use", "size_bytes",
             "uncompressed_bytes", "sha256", "crc32", "crc32_matches_archive", "md5", "matches_zenodo_md5"]
    return table[first + [c for c in table.columns if c not in first]]


def save_plots(ins: Inspector, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, color = "#0b0b0b", "#52514e", "#e5e5e2", "#2a78d6"

    def page(keys, title, filename, shape, xmax):
        fig, axes = plt.subplots(*shape, figsize=(13, 2.45 * shape[0]), constrained_layout=True, sharey=True)
        for ax in axes.ravel():
            ax.set_axis_off()
        for ax, key in zip(axes.ravel(), [k for k in keys if k in ins.traces]):
            item = ins.traces[key]
            ax.set_axis_on()
            for x, y in item["traces"]:
                ax.plot(x, y, color=color, lw=0.6, alpha=0.75)
            ax.set_title(f"House {key[0]} · ch{key[1]} · {item['label']}\n"
                         f"n={item['n']}, median {item['duration']:.0f} min, {item['kwh']:.2f} kWh",
                         fontsize=7.5, color=ink, loc="left")
            ax.set_ylim(0, 3200)
            ax.set_xlim(0, xmax)
            ax.grid(color=grid, lw=0.5)
            ax.tick_params(labelsize=6.5, colors=muted, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle(title, fontsize=10, color=ink, x=0.01, ha="left")
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / filename, dpi=125)
        plt.close(fig)

    wm = sorted(k for k in ins.traces if k not in REVIEW_OTHER)
    page(wm, "Documented washing machine channels: six seeded random events (W vs minutes from start)",
         "review_washing_machine_events.png", (5, 4), 180)
    page(REVIEW_OTHER, "Other channels: dishwashers, dryers, washer-dryers, and channels unlike their label",
         "review_other_channel_events.png", (4, 4), 240)


def run(plot_dir: Path | None = None) -> None:
    """Regenerate the inventory and screening tables from the archives."""
    ins = Inspector(plot_dir)
    for path, handler in ((RAW_ARCHIVE, ins.on_raw), (CLEAN_ARCHIVE, ins.on_clean)):
        with py7zr.SevenZipFile(path, "r") as archive:
            ins.listing.update({f.filename: {"crc32": f.crc32} for f in archive.list()})
        stream_members(path, handler)

    REPORTS.mkdir(parents=True, exist_ok=True)
    screening = build_screening(ins)
    inventory = build_inventory(ins, screening)
    screening.to_csv(REPORTS / "household_screening.csv", index=False)
    inventory.to_csv(REPORTS / "dataset_inventory.csv", index=False)
    if plot_dir:
        save_plots(ins, plot_dir)
    print(f"Wrote {len(inventory)} inventory rows and {len(screening)} screening rows.")
    print(screening.screening_status.value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path, help="Optional folder for manual review plots.")
    run(parser.parse_args().plot_dir)


if __name__ == "__main__":
    main()
