"""Verify or download the official UCI Bike Sharing Dataset (ID 275).

Usage::

    python scripts/download_data.py [--force]

If ``data/raw/hour.csv`` and ``data/raw/day.csv`` already exist with the expected record counts
(17,379 and 731), the script verifies them without hitting the network unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config


def _check_existing() -> bool:
    if not (config.HOUR_CSV.exists() and config.DAY_CSV.exists()):
        return False
    with config.HOUR_CSV.open("r", encoding="utf-8") as fh:
        n_hour = sum(1 for _ in fh) - 1
    with config.DAY_CSV.open("r", encoding="utf-8") as fd:
        n_day = sum(1 for _ in fd) - 1
    return n_hour == config.EXPECTED_HOURLY_ROWS and n_day == config.EXPECTED_DAILY_ROWS


def _stage_from_workspace_root() -> bool:
    root_hour = config.ROOT / "hour.csv"
    root_day = config.ROOT / "day.csv"
    root_readme = config.ROOT / "Readme.txt"
    if root_hour.exists() and root_day.exists():
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root_hour, config.HOUR_CSV)
        shutil.copy2(root_day, config.DAY_CSV)
        if root_readme.exists():
            shutil.copy2(root_readme, config.RAW_DIR / "Readme.txt")
        return _check_existing()
    return False


def download_uci_zip() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official UCI Bike Sharing Dataset from {config.UCI_ZIP_URL} ...")
    try:
        req = urllib.request.Request(
            config.UCI_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (BikeSharingAnalytics/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for member in ("hour.csv", "day.csv", "Readme.txt"):
                if member in zf.namelist():
                    target = config.RAW_DIR / member
                    target.write_bytes(zf.read(member))
        print(f"Extracted hour.csv, day.csv, and Readme.txt to {config.RAW_DIR}")
    except Exception as exc:
        msg = (
            f"Automatic download failed ({exc}).\n"
            f"Manual download instructions:\n"
            f"  1. Visit {config.UCI_URL}\n"
            f"  2. Download 'bike+sharing+dataset.zip'\n"
            f"  3. Extract 'hour.csv' (17,379 rows) and 'day.csv' (731 rows) into:\n"
            f"     {config.RAW_DIR}\n"
        )
        raise RuntimeError(msg) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify or download the UCI Bike Sharing dataset.")
    parser.add_argument("--force", action="store_true", help="Force re-download from UCI.")
    args = parser.parse_args(argv)

    if not args.force:
        if _check_existing():
            print(
                f"Verified existing UCI dataset in {config.RAW_DIR}: "
                f"hour.csv ({config.EXPECTED_HOURLY_ROWS:,} rows), "
                f"day.csv ({config.EXPECTED_DAILY_ROWS:,} rows)."
            )
            return 0
        if _stage_from_workspace_root():
            print(f"Staged verified UCI dataset from workspace root into {config.RAW_DIR}.")
            return 0

    download_uci_zip()
    if not _check_existing():
        raise RuntimeError("Downloaded files did not match expected UCI row counts.")
    print("UCI dataset download and row-count verification succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

