"""Render the cleaning report from the JSON and CSV artefacts, so its numbers cannot drift."""
from __future__ import annotations

import pandas as pd


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Markdown table from a list of dicts, as [(key, heading), ...]."""
    head = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join(" --- " for _ in columns) + "|"
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if value is None or (isinstance(value, float) and value != value):
                cells.append("n/a")
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, int):
                cells.append(f"{value:,}")
            elif isinstance(value, float):
                cells.append(f"{value:,.2f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def _flow(stats: dict) -> str:
    f = stats["row_flow"]
    rows = [
        ("Rows read from Part 1", f["part1_rows_read"]),
        ("Rows read from Part 2", f["part2_rows_read"]),
        ("Part 1 rows not used, at or after the switch", f["part1_rows_not_used_after_switch"]),
        ("Rows after choosing a source", f["rows_after_source_selection"]),
        ("Exact duplicate rows removed from Part 1", f["exact_duplicates_removed_part1"]),
        ("Exact duplicate rows removed from Part 2", f["exact_duplicates_removed_part2"]),
        ("Rows after exact duplicates", f["rows_after_exact_duplicates"]),
        ("Rows merged into another row with the same timestamp", f["rows_merged_into_an_existing_timestamp"]),
        ("Unique timestamps", f["unique_timestamps"]),
        ("Minutes in the output grid", f["minutes_in_grid"]),
        ("Minutes with at least one reading", f["minutes_with_any_reading"]),
    ]
    return table([{"step": a, "count": b} for a, b in rows], [("step", "Step"), ("count", "Count")])


def _raw_files(stats: dict) -> str:
    a, b = stats["raw_files"]["part1"], stats["raw_files"]["part2"]
    keys = [("first_time", "First time"), ("last_time", "Last time"), ("rows", "Rows"),
            ("unique_timestamps", "Unique timestamps"), ("extra_rows_beyond_first", "Rows repeating a timestamp"),
            ("rows_identical_to_an_earlier_row", "Rows identical to an earlier row"),
            ("time_and_unix_disagree_rows", "Rows where Time and Unix differ by 1 s"),
            ("median_positive_gap_s", "Median gap between timestamps (s)"),
            ("gaps_over_2min", "Gaps over 2 min"), ("days_in_gaps_over_2min", "Days in those gaps"),
            ("gaps_over_1d", "Gaps over 1 day"), ("days_in_gaps_over_1d", "Days in those gaps")]
    rows = [{"fact": label, "part1": a[k], "part2": b[k]} for k, label in keys]
    missing = {s["sensor"]: s for s in a["sensors"]}, {s["sensor"]: s for s in b["sensors"]}
    for sensor in missing[0]:
        rows.append({"fact": f"{sensor} missing (%)", "part1": missing[0][sensor]["missing_pct"], "part2": missing[1][sensor]["missing_pct"]})
    return table(rows, [("fact", "Fact"), ("part1", "Part 1"), ("part2", "Part 2")])


def _clock_house1(stats: dict) -> str:
    rows = []
    for item in stats["clock"]["dst_transitions"]:
        for hour in item["hours"]:
            rows.append({"change": f"{item['day']} ({item['direction']})", **hour})
    return table(rows, [("change", "Clock change"), ("utc_hour", "UTC hour"), ("minutes_with_reading", "Minutes with a reading"),
                        ("readings", "Aggregate readings"), ("dst_ambiguous_minutes", "Ambiguous minutes")])


def _reconciliation(recon: dict) -> str:
    parts = []
    names = {"both_use_part1": "Both from Part 1", "ours_part2_official_mostly_part1": "Ours from Part 2, official mostly from Part 1",
             "both_use_part2": "Both from Part 2"}
    columns = [("label", "Sensor"), ("minutes_both", "Minutes in both"), ("minutes_only_ours", "Only ours"),
               ("minutes_only_official", "Only official"), ("r", "r"), ("mean_abs_diff_w", "Mean abs diff (W)"),
               ("ours_mean_w", "Ours mean (W)"), ("official_mean_w", "Official mean (W)")]
    for key, label in names.items():
        period = recon["periods"][key]
        parts.append(f"**{label}** ({period['from']} to {period['to']})\n\n{table(period['sensors'], columns)}")
    return "\n\n".join(parts)


def frozen_summary_rows(frozen: pd.DataFrame) -> list[dict]:
    rows = []
    for house, block in frozen.groupby("house"):
        rows.append({"house": int(house), "periods": len(block), "total_days": round(float(block["days"].sum()), 1),
                     "share_of_minutes_pct": float(block["share_of_house_minutes_pct"].sum().round(1)),
                     "longest_days": round(float(block["days"].max()), 2)})
    return rows


def render(stats: dict, checks: dict, recon: dict, clock_validation: dict, boundaries: pd.DataFrame, frozen: pd.DataFrame,
           manifest: dict) -> str:
    frozen_summary = frozen_summary_rows(frozen)
    wm = "Appliance5"
    conflict_rows, impact_rows, sampling_rows = [], [], []
    for key, label in (("part1", "Part 1"), ("part2", "Part 2")):
        file_checks = checks["duplicates"][key]
        conflict_rows.append({"part": label, **next(e for e in file_checks["neighbour_evidence"] if e["sensor"] == wm)})
        impact_rows.append({"part": label, **next(e for e in file_checks["rule_impact"] if e["sensor"] == wm)})
        sampling_rows.append({"part": label, **file_checks["independent_readings"]})
    nan = checks["part2_nan_behaviour"]
    block_columns = [("official_clock_block", "Official clock"), ("official_minutes", "Minutes"), ("source", "Best source"),
                     ("clean_minus_raw_min", "Official minus raw (min)"), ("r", "r")]
    block_section = "\n\n".join(f"**{name.replace('_', ' ').capitalize()}**\n\n{table(rows, block_columns)}"
                                for name, rows in clock_validation["house1_block_offsets"].items())
    outliers = stats["outliers"]
    review = outliers["aggregate_review"]
    outage = stats["outages"]
    dup = stats["duplicates"]
    boundary = checks["boundary"]
    continuity = boundary["switch_continuity"]
    lag = pd.DataFrame(boundary["daily_lag"])
    drift_days = lag[(lag["part2_lag_min"].abs() >= 1) & (lag["r"] >= 0.8)]
    section = []
    section.append(f"""# House 1 raw cleaning report

Generated by `python -m src.pipeline`. Every number below is read from the pipeline's outputs. DATA.md explains the reasoning.

Run details: Python {manifest['python']}, pandas {manifest['pandas']}, NumPy {manifest['numpy']}. Output table SHA-256 `{manifest['house_1_sha256']}`.

## 1. Row counts

{_flow(stats)}

The switch from Part 1 to Part 2 is at {stats['row_flow']['switch_timestamp_utc']} (UTC).

## 2. Raw files as supplied

{_raw_files(stats)}

## 3. Duplicates and conflicts

{dup['exact_duplicates_removed']:,} rows were exact copies of an earlier row and were removed. A further {dup['rows_merged_into_an_existing_timestamp']:,} rows shared a timestamp with another row and were averaged into it, sensor by sensor.

In the rows used, sensors held conflicting readings {dup['conflicts_logged']:,} times across {dup['timestamps_with_a_conflict']:,} timestamps, touching {dup['minutes_with_a_conflict']:,} minutes. All are in `data/cleaned/house_1_duplicate_conflicts.parquet` with the value each resolved to. Conflicting readings are averaged, so each one still counts.

{table(dup['per_sensor'], [('label', 'Sensor'), ('conflicts', 'Conflicts'), ('zero_vs_nonzero', 'Zero vs non-zero'), ('both_nonzero', 'Both non-zero'), ('involves_spike', 'Involves a spike'), ('spread_100w_or_more', 'Spread 100 W or more'), ('median_spread_w', 'Median spread (W)'), ('max_spread_w', 'Max spread (W)')])}

Evidence that conflicting readings are real changes, measured across each whole file. Rows are in time order if the first reading matches the previous timestamp and the last matches the next.

{table(conflict_rows, [('part', 'File'), ('sensor', 'Sensor'), ('conflicts', 'Conflicts'), ('first_equals_previous_pct', 'First matches previous (%)'), ('last_equals_previous_pct', 'Last matches previous (%)'), ('last_equals_next_pct', 'Last matches next (%)'), ('first_equals_next_pct', 'First matches next (%)')])}

In Part 1, {checks['duplicates']['part1']['conflict_time_split']['with_differing_time_pct']}% of timestamps with a conflict have rows whose `Time` differs under one `Unix` label. In Part 2 the figure is {checks['duplicates']['part2']['conflict_time_split']['with_differing_time_pct']}%.

Effect of the merge rule on the washing machine's 1-minute mean, over each whole file:

{table(impact_rows, [('part', 'File'), ('minutes_with_a_conflict', 'Minutes with a conflict'), ('last_vs_mean_abs_w_p50_in_affected', 'Last vs mean, median (W)'), ('last_vs_mean_abs_w_p99_in_affected', 'Last vs mean, 99th percentile (W)'), ('first_vs_last_abs_w_max', 'First vs last, max (W)'), ('energy_ratio_last_over_mean', 'Energy ratio, last over mean'), ('energy_ratio_first_over_mean', 'Energy ratio, first over mean')])}

## 4. Missing values

Out of {stats['row_flow']['minutes_in_grid']:,} minutes in the grid. Nothing was filled or interpolated.

{table(stats['missing_values'][:-1], [('label', 'Sensor'), ('minutes_present', 'Minutes present'), ('minutes_missing', 'Minutes missing'), ('missing_pct', 'Missing (%)'), ('median_readings_per_minute', 'Median readings per minute'), ('minutes_with_under_3_readings', 'Minutes with under 3 readings'), ('mean_w_when_present', 'Mean when present (W)'), ('energy_kwh_measured', 'Measured energy (kWh)')])}

Readings per minute overstate the independent information in Part 1, which polls in bursts and repeats rows:

{table(sampling_rows, [('part', 'File'), ('rows_per_minute_median', 'Rows per minute'), ('unique_timestamps_per_minute_median', 'Unique timestamps per minute'), ('polls_per_minute_median', 'Independent polls per minute')])}

Part 2 marks a silent sensor with `NaN`. The readme says data was recorded only on a change in load, but `NaN` is about equally likely after an active reading ({nan['nan_rate_after_active_reading_pct']}%) as after an off reading ({nan['nan_rate_after_off_reading_pct']}%), so these look like missed reports. For the washing machine, half of silences last {nan['silence_seconds_p50']:.0f} seconds or less and 99% last {nan['silence_seconds_p99']:.0f} seconds or less:

{table(nan['buckets'], [('length', 'Silence length'), ('silences', 'Silences'), ('hours', 'Hours'), ('same_value_both_sides_pct', 'Same value either side (%)'), ('machine_active_either_side_pct', 'Machine active either side (%)')])}

Zeros in Part 1 are kept even though off and unavailable look the same there:

{table(stats['part1_zeros'], [('sensor', 'Sensor'), ('minutes_with_reading', 'Part 1 minutes with a reading'), ('minutes_exactly_zero', 'Exactly zero'), ('zero_pct', 'Zero (%)')])}

## 5. Outliers

Appliance readings above {outliers['appliance_limit_w']:,} W became missing, the limit the official documentation gives for these monitors.

{table(outliers['appliance_spikes'], [('label', 'Sensor'), ('readings_over_limit', 'Readings removed'), ('part1_readings', 'From Part 1'), ('part2_readings', 'From Part 2'), ('bit_pattern_pct', 'Exact bit patterns (%)'), ('max_w', 'Max (W)'), ('minutes_affected', 'Minutes affected'), ('minutes_left_with_no_reading', 'Minutes left empty')])}

Aggregate readings above {review['threshold_w']:,} W are kept and flagged: {review['readings_over_threshold']:,} readings across {review['minutes_flagged']:,} minutes, of which {review['readings_at_exactly_24000_w']:,} are exactly 24,000 W. {review['isolated_readings']} are isolated single readings that return to normal immediately, which look like sensor errors. Keeping these flagged readings raises the affected minute's mean by a median of {review['effect_on_minute_mean_w']['median']:,} W, and at most {review['effect_on_minute_mean_w']['max']:,} W.

{table(review['episodes'], [('raw_clock_start', 'Raw clock start'), ('kind', 'Kind'), ('rows', 'Readings'), ('max_w', 'Max (W)'), ('reading_before_w', 'Reading before (W)'), ('reading_after_w', 'Reading after (W)'), ('appliance_sum_w', 'Appliance sum (W)')])}

## 6. Outages

Whole-house outages total {outage['whole_house_outage_days']:,} days across {outage['outages_of_any_length']:,} runs. A run of {outage['long_outage_threshold_minutes']} or more empty minutes is flagged `long_outage`. All stay missing.

{table(outage['buckets'], [('length', 'Length'), ('outages', 'Outages'), ('total_days', 'Total days')])}

Longest whole-house outages:

{table(outage['longest'], [('raw_clock_start', 'Raw clock start'), ('days', 'Days'), ('minutes', 'Minutes')])}

Per sensor, including silences while other sensors keep reporting:

{table(outage['per_sensor'], [('sensor', 'Sensor'), ('long_gaps', 'Long gaps'), ('total_days_in_long_gaps', 'Days in long gaps'), ('longest_gap_days', 'Longest gap (days)'), ('missing_minutes_while_other_sensors_report', 'Missing while others report (min)')])}

### Frozen sensors

A sensor repeating one exact minute mean for 60 minutes or more is flagged `<sensor>_frozen`. These values stay in the table, because no cleaning rule removes them yet.

{table(stats['frozen']['per_sensor'], [('label', 'Sensor'), ('runs', 'Runs'), ('minutes', 'Minutes'), ('frozen_energy_kwh', 'Energy in those minutes (kWh)'), ('share_of_sensor_energy_pct', 'Share of sensor energy (%)')])}

Longest frozen periods:

{table(stats['frozen']['longest'], [('sensor', 'Sensor'), ('raw_clock_start', 'Raw clock start'), ('minutes', 'Minutes'), ('value_w', 'Value (W)'), ('source_part', 'Source part')])}

The official cleaned files have the same problem. Aggregate frozen periods for the six households:

{table(frozen_summary, [('house', 'House'), ('periods', 'Frozen periods'), ('total_days', 'Total days'), ('share_of_minutes_pct', 'Share of minutes (%)'), ('longest_days', 'Longest (days)')])}

Every period is listed in `outputs/reports/official_frozen_aggregate_periods.csv`.

## 7. Part 1 and Part 2 boundary

- First Part 2 timestamp: {continuity['first_part2_timestamp']} on the raw clock, {continuity['seconds_between']} seconds after a Part 1 row.
- Step in aggregate at the switch: {continuity['step_at_switch_w']:,} W, against a 95th-percentile minute-to-minute step of {continuity['typical_minute_step_w_p95']:,} W around it.
- The parts overlap for {boundary['overlap_days']} days.

First week of the overlap, by channel:

{table(boundary['channel_agreement_first_week'], [('sensor', 'Sensor'), ('minutes_part1', 'Part 1 minutes'), ('minutes_part2', 'Part 2 minutes'), ('r_same_channel', 'r with same channel'), ('best_part1_channel_for_part2', 'Best-matching Part 1 channel'), ('mean_part1_when_both_w', 'Part 1 mean (W)'), ('mean_part2_when_both_w', 'Part 2 mean (W)')])}

Using Part 2 from the switch gains {boundary['availability_aggregate']['part2_only']:,} aggregate minutes that Part 1 lacks and gives up {boundary['availability_aggregate']['part1_only']:,} that only Part 1 has.

**Clock drift:** on {len(drift_days)} days in the overlap, between {drift_days['day'].min()} and {drift_days['day'].max()}, Part 2 runs later than Part 1 by up to {int(drift_days['part2_lag_min'].max())} minutes. Minutes in the affected window are flagged `clock_uncertain`, {stats['clock']['clock_uncertain_minutes']:,} in total.

Sensors stay aligned within each part, shown here as the lag of the aggregate behind the washing machine:

{table(boundary['within_part_alignment'], [('from', 'From'), ('to', 'To'), ('part1_lag_min', 'Part 1 lag (min)'), ('part1_r', 'Part 1 r'), ('part2_lag_min', 'Part 2 lag (min)'), ('part2_r', 'Part 2 r')])}

Rows per 20 minutes from {boundary['snap_row_rates']['from']} to {boundary['snap_row_rates']['to']}, around the snap-back:

- Part 1: {boundary['snap_row_rates']['part1']}
- Part 2: {boundary['snap_row_rates']['part2']}

Part 1 stays flat while Part 2 wobbles, which suggests Part 2 is the file running late, though this isn't proof.

The drift appears across households, which points to a shared logging cause:

{table(clock_validation['part_drift_by_house'], [('house', 'House'), ('days_compared', 'Days compared'), ('days_with_lag', 'Days with a lag'), ('max_lag_min', 'Max lag (min)'), ('lag_on_2014_08_05', 'Lag on 5 Aug 2014'), ('lag_on_2014_09_10', 'Lag on 10 Sep 2014')])}

## 8. Clock handling

Raw timestamps are UK local clock time. The table holds `timestamp_utc` on a continuous timeline and `raw_clock_minute` with the original clock label.

{_clock_house1(stats)}

The official cleaned files use a clock that changes during the study. Measured per household:

{table(boundaries.to_dict('records'), [('house', 'House'), ('start_day', 'From'), ('end_day', 'To'), ('days', 'Days'), ('clean_minus_raw_min', 'Official minus raw (min)'), ('to_get_uk_local_add_min', 'Add for UK local (min)'), ('status', 'Status'), ('median_r', 'Median r')])}

House 1 at 2-hour resolution shows what happens inside the changes. The official file shifts at raw midnight on clock-change days and whenever its data source changes part:

{block_section}

Days marked `transition` hold a change of offset and are left out of hourly analysis. Days marked `uncertain` have no reliable offset, usually because the aggregate is frozen and has nothing to correlate. A `review` verdict below has that cause for House 10 in spring 2014, when its aggregate was frozen.

Validation of the corrected official clock:

{table(clock_validation['transitions'], [('house', 'House'), ('day', 'Clock change'), ('direction', 'Direction'), ('raw_minutes_by_hour', 'Raw minutes, hours 00 to 03'), ('official_uncorrected_minutes_by_hour', 'Official uncorrected'), ('corrected_minutes_by_hour', 'Corrected'), ('verdict', 'Verdict')])}

Hour 00 of a spring change is not judged, because the official files move it into the previous day, where it interleaves with other rows.

Same-season routine check, in minutes the spring 2015 daily pattern sits later than spring 2014's. A correct correction makes the official clock match raw. This check is coarse, because household routines also change between years, as Houses 8, 15 and 16 show in raw data.

{table(clock_validation['routine_shift'], [('house', 'House'), ('official_uncorrected', 'Official uncorrected'), ('official_corrected', 'Official corrected'), ('raw', 'Raw')])}

## 9. Comparison with the official cleaned file

Compared on UK local labels, using days with a usable clock offset, so "Only ours" includes minutes on days set aside for the official clock. Differences are expected in places: spikes are missing here and zero there, and silent sensors are missing here and held at the last value there.

{_reconciliation(recon)}

Corrected official clock against this table, month by month:

{table(recon['monthly_alignment'], [('month', 'Month'), ('lag_min', 'Best lag (min)'), ('r', 'r'), ('minutes_compared', 'Minutes')])}

Minutes where a spike was removed here:

{table(recon['spike_treatment'], [('sensor', 'Sensor'), ('minutes_with_spike_removed', 'Minutes'), ('ours_missing_in_those_minutes', 'Empty here'), ('official_exactly_zero_minutes', 'Official exactly zero'), ('official_mean_w_in_those_minutes', 'Official mean (W)')])}

Minutes where a sensor is silent here, in the period where both come from Part 2:

{table(recon['held_values_in_official'], [('sensor', 'Sensor'), ('silent_minutes_here', 'Silent minutes'), ('official_has_a_value', 'Official has a value'), ('official_value_is_zero', 'Value is zero'), ('official_value_is_nonzero', 'Value is non-zero'), ('official_energy_kwh_in_those_minutes', 'Official energy (kWh)')])}

Washing machine silences where other sensors keep reporting, and what the official file shows across them:

{table(recon['washing_machine_silence_behaviour'], [('silence_length', 'Silence length'), ('runs', 'Runs'), ('held_one_value', 'Official held one value'), ('held_at_nonzero', 'Held at a non-zero value')])}
""")
    return "\n".join(section)
