"""Run the project pipeline, or any stage on its own.

    python -m src.pipeline                  # every stage, in order
    python -m src.pipeline --stage clean    # one stage

Each stage reads only files on disk and names the stage to run first if an input is
missing. Later project phases add their stages here.
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.analysis import cycles as cycle_detection
from src.analysis import plots as eda_plots
from src.analysis import usage
from src.cleaning import inspect_dataset
from src.cleaning.inspect_dataset import METADATA, parse_metadata
from src.analysis import savings
from src.models import dataset as model_data
from src.models import experiments as model_experiments
from src.models import evaluate as model_eval
from src.models import plots as model_plots
from src.models import train as model_train
from src.cleaning import acquire, clock, plots, prepare, quality, raw_checks, raw_house1, reconcile, report
from src.io_utils import read_archive_member, read_json, read_official_clean, read_raw_part, sha256_file, stream_members, write_json


class MissingArtifact(RuntimeError):
    """Raised when a stage's input does not exist yet."""


def require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise MissingArtifact(f"{path} not found. Run first: python -m src.pipeline --stage {stage}")
    return path


def load_raw_parts(raw_dir: Path = config.RAW_EXTRACT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    return tuple(read_raw_part(require(raw_dir / name, "acquire"), part) for part, name in config.RAW_PART_NAMES.items())


def check_drift_window(daily_lag: list[dict]) -> None:
    """Fail loudly if measured Part 2 drift falls outside the flagged clock window."""
    lo, hi = config.CLOCK_UNCERTAIN_RAW
    table = pd.DataFrame(daily_lag)
    drift = table[(table["part2_lag_min"].abs() >= config.CLOCK_DRIFT_ALERT_MIN) & (table["r"] >= config.CLOCK_DRIFT_MIN_R)]
    outside = drift[(drift["day"] < lo) | (drift["day"] >= hi)]
    if len(outside):
        raise ValueError(f"Part 2 clock drift outside {lo} to {hi}: {outside['day'].tolist()}")


# --------------------------------------------------------------------------- stages
def stage_acquire() -> None:
    result = acquire.run_acquire()
    print(f"Verified {len(result['input_sha256'])} inputs. Raw House 1 files in {config.RAW_EXTRACT_DIR}.")


def stage_inspect() -> None:
    """Regenerate the dataset inventory and household screening from the archives."""
    inspect_dataset.run()


def stage_checks() -> None:
    part1, part2 = load_raw_parts()
    result = raw_checks.run_checks(part1, part2)
    check_drift_window(result["boundary"]["daily_lag"])
    write_json(config.CHECKS_JSON, result)
    print(f"Wrote {config.CHECKS_JSON}.")


def stage_clean(raw_dir: Path = config.RAW_EXTRACT_DIR, parquet: Path = config.HOUSE1_PARQUET,
                conflicts_path: Path = config.CONFLICTS_PARQUET, stats_path: Path = config.CLEANING_JSON) -> None:
    part1, part2 = load_raw_parts(raw_dir)
    result = raw_house1.clean_house1(part1, part2)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    result.minutes.to_parquet(parquet, compression="zstd", index=False)
    result.conflicts.to_parquet(conflicts_path, compression="zstd", index=False)
    write_json(stats_path, quality.build_stats(part1, part2, result))
    print(f"Wrote {parquet}: {len(result.minutes):,} minutes. Conflicts logged: {len(result.conflicts):,}.")


def official_frozen_periods(house: int, clean: pd.Series) -> list[dict]:
    """Periods where an official aggregate sits on one exact value. Measured only."""
    index = clean.index.to_numpy()
    return [{"house": house, "raw_label_start": str(pd.Timestamp(int(index[start]) * 60, unit="s")), "minutes": length,
             "days": round(length / 1440, 3), "value_w": value, "share_of_house_minutes_pct": round(length / len(clean) * 100, 3)}
            for start, length, value in raw_house1.frozen_runs(index, clean.to_numpy())]


def stage_clock() -> None:
    """Measure the official files for each approved household. Nothing is extracted to disk."""
    raw_parts: dict[tuple[int, int], pd.Series] = {}
    clean_series: dict[int, pd.Series] = {}
    names = {clock.raw_member_name(h, p): (h, p) for h in config.HOUSEHOLDS for p in (1, 2)}

    def on_raw(name: str, data: bytes) -> None:
        raw_parts[names[name]] = clock.read_minute_aggregate(data)

    def on_clean(name: str, data: bytes) -> None:
        clean_series[int(name[len("CLEAN_House"):-len(".csv")])] = clock.read_minute_aggregate(data)

    stream_members(config.RAW_ARCHIVE, on_raw, list(names))
    stream_members(config.CLEAN_ARCHIVE, on_clean, [clock.clean_member_name(h) for h in config.HOUSEHOLDS])

    all_daily, all_bounds, transitions, routine, frozen = [], [], [], [], []
    for house in config.HOUSEHOLDS:
        raw = clock.combine_raw(raw_parts[(house, 1)], raw_parts[(house, 2)])
        clean = clean_series[house]
        frozen.extend(official_frozen_periods(house, clean))
        daily = clock.classify_days(clock.daily_offsets(clean, raw))
        daily.insert(0, "house", house)
        all_daily.append(daily)
        all_bounds.append(clock.boundaries(daily, house))
        for item in clock.validate_transitions(clean.index.to_numpy() * 60, raw.index.to_numpy(), daily):
            transitions.append({"house": house, **item})
        routine.append({"house": house, "official_uncorrected": clock.routine_shift(clean),
                        "official_corrected": clock.routine_shift(clock.corrected_minute_series(clean, daily)),
                        "raw": clock.routine_shift(raw)})
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_daily).to_csv(config.CLOCK_DAILY_CSV, index=False)
    pd.concat(all_bounds).to_csv(config.CLOCK_BOUNDARIES_CSV, index=False)
    pd.DataFrame(frozen).to_csv(config.OFFICIAL_FROZEN_CSV, index=False)
    drift = [{"house": house, **clock.part_drift_by_house(raw_parts[(house, 1)], raw_parts[(house, 2)])} for house in config.HOUSEHOLDS]
    blocks = {label: clock.block_offsets(clean_series[1], raw_parts[(1, 1)], raw_parts[(1, 2)], lo, hi)
              for label, lo, hi in (("autumn_2013", "2013-10-26 18:00", "2013-10-27 06:00"),
                                    ("spring_2014", "2014-03-29 18:00", "2014-03-30 06:00"),
                                    ("part1_outage_ends", "2014-09-04 06:00", "2014-09-04 16:00"),
                                    ("part1_ends", "2014-09-30 20:00", "2014-10-01 06:00"))}
    write_json(config.CLOCK_VALIDATION_JSON, {"transitions": transitions, "routine_shift": routine, "part_drift_by_house": drift,
                                              "house1_block_offsets": blocks})
    print(f"Wrote {config.CLOCK_BOUNDARIES_CSV} and validation for {len(config.HOUSEHOLDS)} households.")


def stage_prepare() -> None:
    """Build a 1-minute table per approved household from the official files, in memory."""
    daily = pd.read_csv(require(config.CLOCK_DAILY_CSV, "clock"))
    config.CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    def on_clean(name: str, data: bytes) -> None:
        house = int(name[len("CLEAN_House"):-len(".csv")])
        table = prepare.official_to_minutes(read_official_clean(data), house, daily[daily["house"] == house].reset_index(drop=True))
        table.to_parquet(config.official_parquet(house), compression="zstd", index=False)
        print(f"House {house}: {len(table):,} minutes, {int(table['valid'].sum()):,} valid, {int(table['frozen'].sum()):,} frozen.", flush=True)

    stream_members(config.CLEAN_ARCHIVE, on_clean, [clock.clean_member_name(h) for h in config.HOUSEHOLDS])


def stage_reconcile() -> None:
    minutes = pd.read_parquet(require(config.HOUSE1_PARQUET, "clean"))
    daily = pd.read_csv(require(config.CLOCK_DAILY_CSV, "clock"))
    daily = daily[daily["house"] == 1].reset_index(drop=True)
    official = read_official_clean(read_archive_member(config.CLEAN_ARCHIVE, config.OFFICIAL_HOUSE1_NAME))
    write_json(config.RECONCILIATION_JSON, reconcile.reconcile(minutes, official, daily))
    print(f"Wrote {config.RECONCILIATION_JSON}.")


def stage_plots() -> None:
    part1, part2 = load_raw_parts()
    minutes = pd.read_parquet(require(config.HOUSE1_PARQUET, "clean"))
    for path in plots.make_plots(part1, part2, minutes, config.CLEAN_PLOTS_DIR):
        print(f"Wrote {path}.")


def stage_report() -> None:
    stats = read_json(require(config.CLEANING_JSON, "clean"))
    checks = read_json(require(config.CHECKS_JSON, "checks"))
    recon = read_json(require(config.RECONCILIATION_JSON, "reconcile"))
    validation = read_json(require(config.CLOCK_VALIDATION_JSON, "clock"))
    bounds = pd.read_csv(require(config.CLOCK_BOUNDARIES_CSV, "clock"))
    bounds = bounds.astype(object).where(bounds.notna(), None)
    frozen = pd.read_csv(require(config.OFFICIAL_FROZEN_CSV, "clock"))
    manifest = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "house_1_sha256": sha256_file(require(config.HOUSE1_PARQUET, "clean"))}
    config.CLEANING_MD.write_text(report.render(stats, checks, recon, validation, bounds, frozen, manifest), encoding="utf-8")
    print(f"Wrote {config.CLEANING_MD}.")


def load_prepared(houses=None) -> dict[int, pd.DataFrame]:
    """Prepared 1-minute tables, in the approved household order."""
    return {house: pd.read_parquet(require(config.official_parquet(house), "prepare")) for house in (houses or config.HOUSEHOLDS)}


def stage_eda() -> None:
    """Washing machine analysis across the six households. Tables and plots go to outputs."""
    tables = load_prepared()
    reports, plots_dir = config.EDA_REPORTS_DIR, config.EDA_PLOTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    occupancy = parse_metadata(METADATA)[0].set_index("house")["occupancy"]

    all_cycles, quality_rows, state_rows, summary_rows, hourly, weekday, clock_rows, frozen_rows, parameter_rows, examples, days = ([] for _ in range(11))
    for house, table in tables.items():
        found = usage.household_cycles(table, house)
        all_cycles.append(found)
        quality_rows.append(usage.quality_summary(table, house))
        state_rows.append(usage.power_states(table, house))
        summary_rows.append({**usage.cycle_summary(found, table, house), "occupancy": int(occupancy[house])})
        hourly.append(usage.hourly_usage(found, table, house))
        weekday.append(usage.weekday_usage(found, table, house))
        clock_rows.append(usage.clock_sensitivity(found, table, house))
        frozen_rows.append(usage.frozen_sensitivity(table, house))
        parameter_rows.append(usage.parameter_sensitivity(table, house))
        examples.append(usage.example_cycles(found))
        days.append((house, usage.representative_day(found, table)))

    cycles = pd.concat(all_cycles, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    hourly_table, weekday_table = pd.concat(hourly, ignore_index=True), pd.concat(weekday, ignore_index=True)
    cycles.to_csv(reports / "cycles.csv", index=False)
    summary.to_csv(reports / "cycle_summary.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(reports / "data_quality.csv", index=False)
    pd.DataFrame(state_rows).to_csv(reports / "power_states.csv", index=False)
    hourly_table.to_csv(reports / "hourly_usage.csv", index=False)
    weekday_table.to_csv(reports / "weekday_usage.csv", index=False)
    pd.DataFrame(clock_rows).to_csv(reports / "clock_sensitivity.csv", index=False)
    pd.DataFrame(frozen_rows).to_csv(reports / "frozen_sensitivity.csv", index=False)
    pd.concat(parameter_rows, ignore_index=True).to_csv(reports / "parameter_sensitivity.csv", index=False)
    write_json(reports / "cycle_parameters.json", cycle_detection.params_dict())

    chosen_days = {house: day for house, day in days if day is not None}
    pd.DataFrame([{"house": h, "day": str(d.date())} for h, d in chosen_days.items()]).to_csv(reports / "representative_days.csv", index=False)
    for path in (eda_plots.representative_traces(chosen_days, tables, plots_dir / "representative_days.png"),
                 eda_plots.cycle_examples(pd.concat(examples), tables, plots_dir / "detected_cycles.png"),
                 eda_plots.hourly_distribution(hourly_table, plots_dir / "hourly_distribution.png"),
                 eda_plots.weekday_distribution(weekday_table, plots_dir / "weekday_distribution.png"),
                 eda_plots.cross_household(summary, plots_dir / "cross_household.png"),
                 eda_plots.duration_energy(cycles, plots_dir / "duration_energy.png")):
        print(f"Wrote {path}.")
    print(f"Detected {len(cycles):,} cycles across {len(tables)} households.")


def load_model_data(tables: dict[int, pd.DataFrame], **options) -> dict[int, model_data.HouseData]:
    return {house: model_data.build_house(table, house, **options) for house, table in tables.items()}


def stage_train() -> None:
    """Leave-one-household-out Random Forest. Choices use training houses only."""
    tables = load_prepared()
    data = load_model_data(tables)
    predictions_dir = config.MODELS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    config.MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log, selection_rows, importance = [], [], {}
    for fold, (house, test) in enumerate(data.items()):
        train = [d for h, d in data.items() if h != house]
        seed = config.SEED + fold
        selection, off_per_active, threshold = model_train.select_on_training_houses(train, seed)
        selection_rows.append(selection.assign(held_out_house=house))
        forest, predicted, info = model_train.run_fold(test, train, model_train.FINAL, off_per_active, seed)
        table = tables[house]
        frame = pd.DataFrame({
            "grid_pos": test.positions, "official_minute": table["official_minute"].to_numpy()[test.positions],
            "uk_local_minute": table["uk_local_minute"].to_numpy()[test.positions], "aggregate_w": table["aggregate_w"].to_numpy()[test.positions],
            "actual_w": test.y, "predicted_w": predicted, "threshold_w": threshold, "issues_any": table["issues_any"].to_numpy()[test.positions]})
        frame.to_parquet(predictions_dir / f"house_{house}.parquet", compression="zstd", index=False)
        model_train.save_model(forest, config.MODELS_DIR / f"random_forest_holdout_house_{house}.joblib")
        importance[house] = model_train.impurity_importance(forest, test.X.columns)
        log.append({"held_out_house": house, "training_houses": [d.house for d in train], "chosen_off_per_active": off_per_active,
                    "threshold_w": threshold, "evaluated_minutes": len(test.y), **info})
        print(f"House {house} held out: ratio {off_per_active}, threshold {threshold} W, fit {info['fit_seconds']} s, MAE {float(np.abs(predicted - test.y).mean()):.1f} W.", flush=True)
    pd.concat(selection_rows, ignore_index=True).to_csv(config.MODEL_REPORTS_DIR / "inner_selection.csv", index=False)
    write_json(config.MODEL_REPORTS_DIR / "training_log.json", {"settings": model_train.settings_dict(), "folds": log, "feature_importance": importance})


def load_predictions(directory: Path | None = None) -> dict[int, pd.DataFrame]:
    folder = directory or config.MODELS_DIR / "predictions"
    return {house: pd.read_parquet(require(folder / f"house_{house}.parquet", "train")) for house in config.HOUSEHOLDS}


def appliance_names() -> dict[int, dict[str, str]]:
    """Documented appliance names for the other channels in each prepared table."""
    excel = parse_metadata(METADATA)[1]
    return {house: {f"other_appliance{channel}_w": excel[(house, channel)]["label"] for channel in range(1, 10)} for house in config.HOUSEHOLDS}


def stage_evaluate() -> None:
    """Metrics, error analysis and plots for the held-out predictions."""
    predictions = load_predictions()
    tables = load_prepared()
    grids = {house: len(table) for house, table in tables.items()}
    reports, plots_dir = config.MODEL_REPORTS_DIR, config.MODEL_PLOTS_DIR
    reports.mkdir(parents=True, exist_ok=True)

    metrics = model_eval.metric_tables(predictions)
    metrics.to_csv(reports / "metrics.csv", index=False)
    model_eval.macro_means(metrics).to_csv(reports / "metrics_macro_mean.csv", index=False)
    cycle_table, details = model_eval.cycle_performance(predictions, grids)
    cycle_table.to_csv(reports / "cycle_detection.csv", index=False)
    model_eval.missed_cycle_analysis(details).to_csv(reports / "cycle_recall_by_type.csv", index=False)
    model_eval.error_sources(predictions).to_csv(reports / "error_sources.csv", index=False)
    model_eval.error_by_aggregate_level(predictions).to_csv(reports / "error_by_aggregate_level.csv", index=False)
    names = appliance_names()
    model_eval.false_positive_sources(predictions, tables, names).to_csv(reports / "false_alarm_sources.csv", index=False)
    model_eval.practical_estimates(predictions).to_csv(reports / "practical_estimates.csv", index=False)
    cases = model_eval.failure_cases(predictions, grids, details, tables, names)
    pd.DataFrame(cases).drop(columns="title").to_csv(reports / "failure_cases.csv", index=False)

    log = read_json(require(reports / "training_log.json", "train"))
    importance = pd.DataFrame(log["feature_importance"]).mean(axis=1)
    importance.sort_values(ascending=False).rename("importance").to_csv(reports / "feature_importance.csv", index_label="feature")
    chosen = {house: model_eval.typical_cycle(details, house) for house in predictions}
    for path in (model_plots.actual_vs_predicted(chosen, predictions, grids, plots_dir / "actual_vs_predicted.png"),
                 model_plots.scatter_density(predictions, plots_dir / "predicted_against_actual.png"),
                 model_plots.model_vs_baseline(metrics, plots_dir / "model_vs_baseline.png"),
                 model_plots.feature_importance(importance, plots_dir / "feature_importance.png"),
                 model_plots.failure_cases(cases, predictions, grids, plots_dir / "failure_cases.png")):
        print(f"Wrote {path}.")


def stage_experiments() -> None:
    """Secondary experiments. None of them replaces the primary leave-one-house-out result."""
    tables = load_prepared()
    primary = load_predictions()
    grids = {house: len(table) for house, table in tables.items()}
    log = read_json(require(config.MODEL_REPORTS_DIR / "training_log.json", "train"))
    folds = {fold["held_out_house"]: fold for fold in log["folds"]}
    seed = config.SEED
    rows = model_experiments.summarise("primary", primary, grids=grids, label="Primary: 1-minute, trailing features")

    print("Experiment: frozen minutes kept.", flush=True)
    kept, _ = model_experiments.run_forest_variant(tables, folds, seed, exclude_frozen=False)
    rows += model_experiments.summarise("no_frozen_exclusion", kept, grids=grids, label="Frozen minutes kept in training and evaluation")
    rows += model_experiments.compare_on_common_minutes(primary, kept)

    print("Experiment: retrospective features.", flush=True)
    retrospective, _ = model_experiments.run_forest_variant(tables, folds, seed, reselect_threshold=True, retrospective=True)
    rows += model_experiments.summarise("retrospective", retrospective, grids=grids, label="Retrospective: future aggregate readings allowed")

    print("Experiment: two-stage model.", flush=True)
    two_stage, two_stage_log = model_experiments.run_two_stage(tables, seed)
    rows += model_experiments.summarise("two_stage", two_stage, grids=grids, label="Two-stage: classifier then regressor")

    extra_log = {"two_stage": two_stage_log}
    for step in (15, 30):
        print(f"Experiment: {step}-minute readings.", flush=True)
        coarse, coarse_log = model_experiments.run_forest_variant(model_experiments.resolution_tables(tables, step), folds, seed, reselect_all=True, step_minutes=step)
        rows += model_experiments.summarise(f"interval_{step}min", coarse, step, label=f"Trained on {step}-minute readings")
        rows += model_experiments.summarise(f"primary_averaged_{step}min", model_experiments.aggregate_primary_to_interval(primary, step), step, label=f"1-minute model averaged to {step} minutes")
        extra_log[f"interval_{step}min"] = coarse_log
    config.MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.MODEL_REPORTS_DIR / "experiments.csv", index=False)
    model_experiments.signature_visibility(tables).to_csv(config.MODEL_REPORTS_DIR / "signature_visibility.csv", index=False)
    write_json(config.MODEL_REPORTS_DIR / "experiments_log.json", extra_log)
    print("Wrote experiment tables.")


def stage_recommend() -> None:
    """Size the opportunity and its measurement, from measured cycles and model results."""
    tables = load_prepared()
    eda = config.EDA_REPORTS_DIR
    summary = pd.read_csv(require(eda / "cycle_summary.csv", "eda"))
    cycles = pd.read_csv(require(eda / "cycles.csv", "eda"))
    states = pd.read_csv(require(eda / "power_states.csv", "eda"))
    experiments = pd.read_csv(require(config.MODEL_REPORTS_DIR / "experiments.csv", "experiments"))
    visibility = pd.read_csv(require(config.MODEL_REPORTS_DIR / "signature_visibility.csv", "experiments"))
    reports, plots_dir = config.RECOMMENDATION_REPORTS_DIR, config.RECOMMENDATION_PLOTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    opportunity = savings.opportunity_table(summary, cycles, tables, states)
    opportunity.to_csv(reports / "opportunity.csv", index=False)
    savings.measurement_power(opportunity).to_csv(reports / "measurement_power.csv", index=False)
    house = 10
    own = cycles[(cycles["house"] == house) & cycles["complete"]]
    typical = own.assign(distance=(own["duration_min"] - own["duration_min"].median()).abs()).sort_values(["distance", "start_pos"]).iloc[0]
    for path in (eda_plots.saving_by_household(opportunity, plots_dir / "saving_by_household.png"),
                 eda_plots.resolution(tables[house], typical, house, experiments, visibility, plots_dir / "resolution.png")):
        print(f"Wrote {path}.")


def stage_results() -> None:
    """Write RESULTS.md and Recommendation.md from the saved outputs."""
    from src import reporting
    for path in reporting.write_documents():
        print(f"Wrote {path}.")


def stage_notebooks() -> None:
    """Write and execute the four notebooks, which display the pipeline's saved outputs."""
    from src import build_notebooks
    for path in build_notebooks.build_and_execute():
        print(f"Executed {path}.")


STAGES = {
    "acquire": stage_acquire,
    "inspect": stage_inspect,
    "checks": stage_checks,
    "clean": stage_clean,
    "clock": stage_clock,
    "prepare": stage_prepare,
    "reconcile": stage_reconcile,
    "plots": stage_plots,
    "report": stage_report,
    "eda": stage_eda,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "experiments": stage_experiments,
    "recommend": stage_recommend,
    "results": stage_results,
    "notebooks": stage_notebooks,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="REFIT cleaning pipeline.")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all", help="Stage to run. The default runs all of them.")
    args = parser.parse_args(argv)
    for name, stage in STAGES.items():
        if args.stage in (name, "all"):
            print(f"--- {name} ---", flush=True)
            stage()


if __name__ == "__main__":
    main()
