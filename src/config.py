"""Single source of truth for paths, identifiers and thresholds."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# Inputs
ARCHIVE_DIR = ROOT / "data" / "raw" / "archives"
DOCS_DIR = ROOT / "data" / "raw" / "docs"
RAW_EXTRACT_DIR = ROOT / "data" / "raw" / "refit_raw"
RAW_ARCHIVE = ARCHIVE_DIR / "REFIT_RAW_081116.7z"
CLEAN_ARCHIVE = ARCHIVE_DIR / "CLEAN_REFIT_081116.7z"
RAW_PART_NAMES = {1: "RAW_House1_Part1.csv", 2: "RAW_House1_Part2.csv"}
OFFICIAL_HOUSE1_NAME = "CLEAN_House1.csv"

# SHA-256 of each downloaded file, recorded in Phase 2. A mismatch stops the pipeline.
INPUT_SHA256 = {
    RAW_ARCHIVE: "f294fe673532220dc527b5e7d7a25e99568bd24acf9fb5be0a0def14af64b2cf",
    CLEAN_ARCHIVE: "4dc4dde763da1c843e9d565fc68f1a63752d858980615635994d7f701fdfcc4b",
    DOCS_DIR / "RAW_READ_ME_081116.txt": "6b20b821a36af681e460bde5d24b933aa58e191c4c3f8de05218941e6d34937d",
    DOCS_DIR / "CLEAN_READ_ME_081116.txt": "c5f5236fd5a7b29771913480fcef6e9be8a85d07511939c5a35160b7afc97bfb",
    DOCS_DIR / "MetaData_Tables.xlsx": "3d071013b178409e543d558e837ef806f1f2b108f665ab78e015a4411519fccb",
}

# Outputs
CLEANED_DIR = ROOT / "data" / "cleaned"
REPORTS_DIR = ROOT / "outputs" / "reports"
PLOTS_DIR = ROOT / "outputs" / "plots"
CLEAN_PLOTS_DIR = PLOTS_DIR / "cleaning"
HOUSE1_PARQUET = CLEANED_DIR / "house_1_1min.parquet"
CONFLICTS_PARQUET = CLEANED_DIR / "house_1_duplicate_conflicts.parquet"
CHECKS_JSON = REPORTS_DIR / "raw_house1_checks.json"
CLEANING_JSON = REPORTS_DIR / "cleaning_report.json"
CLEANING_MD = REPORTS_DIR / "cleaning_report.md"
CLOCK_BOUNDARIES_CSV = REPORTS_DIR / "clock_boundaries.csv"
CLOCK_DAILY_CSV = REPORTS_DIR / "clock_daily_offsets.csv"
CLOCK_VALIDATION_JSON = REPORTS_DIR / "clock_validation.json"
RECONCILIATION_JSON = REPORTS_DIR / "reconciliation.json"
OFFICIAL_FROZEN_CSV = REPORTS_DIR / "official_frozen_aggregate_periods.csv"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
EDA_PLOTS_DIR = PLOTS_DIR / "eda"
MODEL_REPORTS_DIR = REPORTS_DIR / "model"
MODEL_PLOTS_DIR = PLOTS_DIR / "model"
RECOMMENDATION_REPORTS_DIR = REPORTS_DIR / "recommendation"
RECOMMENDATION_PLOTS_DIR = PLOTS_DIR / "recommendation"
MODELS_DIR = ROOT / "outputs" / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"


def official_parquet(house: int) -> Path:
    """Prepared 1-minute table for one household, from the official cleaned file."""
    return CLEANED_DIR / f"official_house_{house}_1min.parquet"

# Dataset structure, from RAW_READ_ME_081116.txt and checked against the data.
APPLIANCES = [f"Appliance{i}" for i in range(1, 10)]
SENSORS = ["Aggregate", *APPLIANCES]
RAW_COLUMNS = ["Time", "Unix", *SENSORS]
HOUSE1_LABELS = {
    "Aggregate": "Aggregate",
    "Appliance1": "Fridge",
    "Appliance2": "Freezer (1)",
    "Appliance3": "Freezer (2)",
    "Appliance4": "Washer dryer",
    "Appliance5": "Washing machine",
    "Appliance6": "Dishwasher",
    "Appliance7": "Computer",
    "Appliance8": "Television site",
    "Appliance9": "Electric heater",
}

# Households approved after Phase 2, as {house: documented washing machine channel}.
HOUSEHOLDS = {1: 5, 2: 2, 8: 4, 10: 5, 15: 3, 16: 5}

# Cleaning thresholds. DATA.md records the evidence behind each one.
APPLIANCE_LIMIT_W = 4000  # Official rated limit of the appliance monitors.
AGGREGATE_REVIEW_W = 23000  # Flagged for investigation and never deleted.
LONG_OUTAGE_MINUTES = 3  # A run of this many empty minutes is at least a 2-minute silence.
# A sensor repeating one exact minute mean this long is treated as frozen. Flagged, never removed.
FROZEN_MIN_MINUTES = 60
FROZEN_MIN_APPLIANCE_W = 20  # Below this an appliance is off or idle, where a flat reading is normal.
# Raw clock window in which Part 2 runs up to 7 minutes later than Part 1 (end exclusive).
CLOCK_UNCERTAIN_RAW = ("2014-07-11", "2014-09-04")
CLOCK_DRIFT_ALERT_MIN = 2  # A daily Part 2 lag this large must fall inside the window above.
CLOCK_DRIFT_MIN_R = 0.8
TIMEZONE = "Europe/London"
DST_TRANSITIONS = {  # Raw clock date and direction of each UK change inside the study.
    "2013-10-27": "back",
    "2014-03-30": "forward",
    "2014-10-26": "back",
    "2015-03-29": "forward",
}
