"""Central paths and constants used across the project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = ROOT / "reports"

HOUR_CSV = RAW_DIR / "hour.csv"
DAY_CSV = RAW_DIR / "day.csv"

EXPECTED_HOURLY_ROWS = 17_379
EXPECTED_DAILY_ROWS = 731

UCI_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

DATASET_CITATION = (
    "Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5W894. Original publication: Fanaee-T, H. and Gama, J. (2013), "
    "'Event labeling combining ensemble detectors and background knowledge', "
    "Progress in Artificial Intelligence, pp. 1-15, doi:10.1007/s13748-013-0040-3."
)

# Pre-registered commuting windows (inclusive hour-of-day values; hr=7 means 07:00-07:59).
MORNING_COMMUTE_HOURS = (7, 8)          # 07:00-08:59
EVENING_COMMUTE_HOURS = (16, 17, 18)    # 16:00-18:59
COMMUTE_HOURS = MORNING_COMMUTE_HOURS + EVENING_COMMUTE_HOURS

RANDOM_STATE = 42


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

