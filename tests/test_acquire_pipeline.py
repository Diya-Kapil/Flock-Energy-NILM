"""Input checks, selective extraction, stage independence, and an end-to-end run on disk."""
from __future__ import annotations

import pandas as pd
import py7zr
import pytest

from conftest import csv_text, make_part, steady_rows
from src import pipeline
from src.cleaning import acquire
from src.config import RAW_PART_NAMES
from src.io_utils import SchemaError, read_official_clean, read_raw_part, sha256_file, stream_members


@pytest.fixture
def archive(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    for name in ("wanted.csv", "other.csv"):
        (folder / name).write_text(f"contents of {name}\n")
    path = tmp_path / "test.7z"
    with py7zr.SevenZipFile(path, "w") as handle:
        handle.write(folder / "wanted.csv", "wanted.csv")
        handle.write(folder / "other.csv", "other.csv")
    return path


class TestVerifyInputs:
    def test_matching_checksum_passes(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("data")
        assert acquire.verify_inputs({path: sha256_file(path)}) == {"f.txt": sha256_file(path)}

    def test_changed_file_stops_the_run(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("data")
        with pytest.raises(acquire.InputError, match="Checksum mismatch"):
            acquire.verify_inputs({path: "0" * 64})

    def test_missing_file_says_what_to_do(self, tmp_path):
        with pytest.raises(acquire.InputError, match="DATA.md"):
            acquire.verify_inputs({tmp_path / "nope.7z": "0" * 64})


class TestExtraction:
    def test_extracts_only_named_members(self, archive, tmp_path):
        out = tmp_path / "out"
        report = acquire.extract_members(archive, ["wanted.csv"], out)
        assert [p.name for p in out.iterdir()] == ["wanted.csv"]
        assert report["wanted.csv"]["extracted_this_run"]

    def test_archive_is_not_modified(self, archive, tmp_path):
        before = sha256_file(archive), archive.stat().st_mtime_ns
        acquire.extract_members(archive, ["wanted.csv"], tmp_path / "out")
        assert (sha256_file(archive), archive.stat().st_mtime_ns) == before

    def test_second_run_reuses_a_good_file(self, archive, tmp_path):
        acquire.extract_members(archive, ["wanted.csv"], tmp_path / "out")
        assert not acquire.extract_members(archive, ["wanted.csv"], tmp_path / "out")["wanted.csv"]["extracted_this_run"]

    def test_corrupt_extracted_file_is_replaced(self, archive, tmp_path):
        out = tmp_path / "out"
        acquire.extract_members(archive, ["wanted.csv"], out)
        (out / "wanted.csv").write_text("tampered")
        report = acquire.extract_members(archive, ["wanted.csv"], out)
        assert report["wanted.csv"]["extracted_this_run"]
        assert (out / "wanted.csv").read_text() == "contents of wanted.csv\n"

    def test_unknown_member_is_an_error(self, archive, tmp_path):
        with pytest.raises(acquire.InputError, match="does not contain"):
            acquire.extract_members(archive, ["missing.csv"], tmp_path / "out")


class TestStreaming:
    def test_members_stay_in_memory(self, archive, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        before = sorted(p.name for p in tmp_path.iterdir())
        seen = {}
        stream_members(archive, lambda name, data: seen.update({name: data}), ["other.csv"])
        assert seen == {"other.csv": b"contents of other.csv\n"}
        assert sorted(p.name for p in tmp_path.iterdir()) == before


class TestSchema:
    def test_wrong_columns_are_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Time,Unix,Power\n2014-01-01 00:00:00,1,2\n")
        with pytest.raises(SchemaError):
            read_raw_part(path, 1)

    def test_official_file_needs_issues_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Time,Unix,Power\n2014-01-01 00:00:00,1,2\n")
        with pytest.raises(SchemaError):
            read_official_clean(path)

    def test_nan_text_is_read_as_missing(self, tmp_path):
        frame = make_part([{"clock": "2014-01-10 12:00:00", "Appliance5": None}], 2)
        path = tmp_path / "part.csv"
        path.write_text(csv_text(frame))
        assert read_raw_part(path, 2)["Appliance5"].isna().all()

    def test_file_order_is_recorded(self, tmp_path):
        frame = make_part([{"clock": "2014-01-10 12:00:10"}, {"clock": "2014-01-10 12:00:00"}], 1)
        path = tmp_path / "part.csv"
        path.write_text(csv_text(frame))
        assert read_raw_part(path, 1)["file_row"].tolist() == [0, 1]


class TestStageIndependence:
    def test_missing_input_names_the_stage_to_run(self, tmp_path):
        with pytest.raises(pipeline.MissingArtifact, match="--stage acquire"):
            pipeline.load_raw_parts(tmp_path)

    def test_require_passes_for_existing_file(self, tmp_path):
        path = tmp_path / "x"
        path.write_text("y")
        assert pipeline.require(path, "clean") == path

    @pytest.mark.parametrize("stage", ["reconcile", "report"])
    def test_later_stages_fail_clearly_without_inputs(self, stage, monkeypatch, tmp_path):
        for name in ("HOUSE1_PARQUET", "CLOCK_DAILY_CSV", "CLEANING_JSON", "CHECKS_JSON", "RECONCILIATION_JSON",
                     "CLOCK_VALIDATION_JSON", "CLOCK_BOUNDARIES_CSV", "OFFICIAL_FROZEN_CSV"):
            monkeypatch.setattr(pipeline.config, name, tmp_path / f"{name}.missing")
        with pytest.raises(pipeline.MissingArtifact, match="--stage"):
            pipeline.STAGES[stage]()

    def test_every_stage_is_selectable(self):
        assert {"acquire", "inspect", "checks", "clean", "clock", "prepare", "reconcile", "plots", "report", "eda", "train",
                "evaluate", "experiments", "recommend", "results", "notebooks"} == set(pipeline.STAGES)

    def test_drift_outside_flagged_window_stops_the_run(self):
        with pytest.raises(ValueError, match="clock drift"):
            pipeline.check_drift_window([{"day": "2015-01-01", "part2_lag_min": 5, "r": 0.99}])

    def test_drift_inside_flagged_window_is_accepted(self):
        pipeline.check_drift_window([{"day": "2014-08-01", "part2_lag_min": 5, "r": 0.99}])


class TestEndToEnd:
    @pytest.fixture
    def cleaned(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        part1 = make_part(steady_rows("2014-01-10 12:00", 30, Appliance5=100), 1)
        part2 = make_part(steady_rows("2014-01-10 12:20", 30, Appliance5=100), 2)
        (raw / RAW_PART_NAMES[1]).write_text(csv_text(part1))
        (raw / RAW_PART_NAMES[2]).write_text(csv_text(part2))
        paths = {"parquet": tmp_path / "out.parquet", "conflicts_path": tmp_path / "c.parquet", "stats_path": tmp_path / "s.json"}
        pipeline.stage_clean(raw_dir=raw, **paths)
        return raw, paths

    def test_stage_writes_parquet_with_expected_columns(self, cleaned):
        _, paths = cleaned
        table = pd.read_parquet(paths["parquet"])
        for column in ("timestamp_utc", "raw_clock_minute", "source_part", "Aggregate", "Aggregate_n", "Aggregate_missing",
                       "Aggregate_gap_minutes", "Aggregate_frozen", "Appliance5_spikes", "Aggregate_over_23kw", "n_conflicts",
                       "outage_minutes", "long_outage", "dst_ambiguous", "clock_uncertain"):
            assert column in table.columns
        assert len(table) == 50

    def test_rerun_gives_an_identical_file(self, cleaned, tmp_path):
        raw, paths = cleaned
        again = tmp_path / "again.parquet"
        pipeline.stage_clean(raw_dir=raw, parquet=again, conflicts_path=tmp_path / "c2.parquet", stats_path=tmp_path / "s2.json")
        assert sha256_file(again) == sha256_file(paths["parquet"])

    def test_parquet_round_trips_types(self, cleaned):
        _, paths = cleaned
        table = pd.read_parquet(paths["parquet"])
        assert str(table["timestamp_utc"].dtype).endswith("UTC]")
        assert table["Aggregate_missing"].dtype == bool
        assert str(table["source_part"].dtype) == "Int8"
