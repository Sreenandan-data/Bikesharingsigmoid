"""Source-data validation for the UCI Bike Sharing Dataset.

Run as a script to write ``reports/data_quality_report.md`` and
``outputs/data_quality.json``::

    python -m src.data_validation
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

HOURLY_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]
DAILY_COLUMNS = [c for c in HOURLY_COLUMNS if c != "hr"]

# Valid domains from the official README.
CATEGORICAL_DOMAINS = {
    "season": {1, 2, 3, 4},
    "yr": {0, 1},
    "mnth": set(range(1, 13)),
    "hr": set(range(24)),
    "holiday": {0, 1},
    "weekday": set(range(7)),
    "workingday": {0, 1},
    "weathersit": {1, 2, 3, 4},
}
NORMALISED_COLUMNS = ["temp", "atemp", "hum", "windspeed"]
COUNT_COLUMNS = ["casual", "registered", "cnt"]


class DataNotFoundError(FileNotFoundError):
    """Raised when the source CSV files are missing."""


def load_raw(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(
            f"Source file not found: {path}. Run `python scripts/download_data.py` or download "
            f"the dataset manually from {config.UCI_URL} and extract hour.csv/day.csv into "
            f"{config.RAW_DIR}."
        )
    return pd.read_csv(path)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" (data defect) or "info" (worth knowing, not a defect)

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": bool(self.passed), "detail": self.detail,
                "severity": self.severity}


@dataclass
class ValidationReport:
    dataset: str
    n_rows: int
    n_cols: int
    checks: list[CheckResult] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def add(self, name: str, passed: bool, detail: str, severity: str = "error") -> None:
        self.checks.append(CheckResult(name, bool(passed), detail, severity))

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "n_rows": self.n_rows, "n_cols": self.n_cols,
                "passed": self.passed, "checks": [c.as_dict() for c in self.checks],
                "extras": self.extras}


def _common_checks(df: pd.DataFrame, report: ValidationReport, expected_cols: list[str],
                   expected_rows: int) -> None:
    report.add("columns", list(df.columns) == expected_cols,
               f"Observed columns: {list(df.columns)}")
    report.add("row_count", len(df) == expected_rows,
               f"Observed {len(df)} rows; README documents {expected_rows}.")
    missing = df.isna().sum()
    report.add("missing_values", int(missing.sum()) == 0,
               "No missing values." if missing.sum() == 0
               else f"Missing values per column: {missing[missing > 0].to_dict()}")
    report.extras["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    n_dup = int(df.duplicated().sum())
    report.add("exact_duplicate_rows", n_dup == 0, f"{n_dup} exact duplicate rows.")
    n_dup_id = int(df["instant"].duplicated().sum())
    report.add("duplicate_instant", n_dup_id == 0, f"{n_dup_id} duplicate `instant` identifiers.")
    key = ["dteday", "hr"] if "hr" in df.columns else ["dteday"]
    n_dup_key = int(df.duplicated(subset=key).sum())
    report.add("duplicate_time_key", n_dup_key == 0, f"{n_dup_key} duplicate {key} keys.")

    parsed = pd.to_datetime(df["dteday"], format="%Y-%m-%d", errors="coerce")
    n_bad = int(parsed.isna().sum())
    report.add("date_parsing", n_bad == 0, f"{n_bad} unparseable dates; range "
               f"{parsed.min().date()} to {parsed.max().date()}.")

    sort_key = parsed + pd.to_timedelta(df["hr"], unit="h") if "hr" in df.columns else parsed
    report.add("chronological_order", bool(sort_key.is_monotonic_increasing),
               "Records are in chronological order." if sort_key.is_monotonic_increasing
               else "Records are NOT in chronological order.")
    report.add("instant_sequential", bool((df["instant"].diff().dropna() == 1).all()),
               "`instant` increments by 1 for every row.")

    for col, domain in CATEGORICAL_DOMAINS.items():
        if col not in df.columns:
            continue
        bad = sorted(set(df[col].unique()) - domain)
        report.add(f"domain_{col}", not bad,
                   f"Observed values {sorted(df[col].unique().tolist())}; invalid: {bad}")

    for col in NORMALISED_COLUMNS:
        lo, hi = float(df[col].min()), float(df[col].max())
        report.add(f"range_{col}", 0.0 <= lo and hi <= 1.0,
                   f"min={lo:.4f}, max={hi:.4f} (expected within [0, 1]).")

    for col in COUNT_COLUMNS:
        n_neg = int((df[col] < 0).sum())
        report.add(f"nonnegative_{col}", n_neg == 0, f"{n_neg} negative values in `{col}`.")
        report.add(f"integer_{col}", bool(np.all(np.equal(np.mod(df[col], 1), 0))),
                   f"`{col}` contains only whole numbers.")

    mismatch = int((df["cnt"] != df["casual"] + df["registered"]).sum())
    report.add("cnt_equals_components", mismatch == 0,
               f"{mismatch} rows where cnt != casual + registered.")

    # Calendar consistency derived from the date itself.
    report.add("year_consistent", bool((parsed.dt.year - 2011 == df["yr"]).all()),
               "`yr` matches the parsed date year (0=2011, 1=2012).")
    report.add("month_consistent", bool((parsed.dt.month == df["mnth"]).all()),
               "`mnth` matches the parsed date month.")
    # Python: Monday=0..Sunday=6; dataset: Sunday=0..Saturday=6.
    report.add("weekday_consistent", bool(((parsed.dt.day_of_week + 1) % 7 == df["weekday"]).all()),
               "`weekday` matches the parsed date (0=Sunday ... 6=Saturday).")
    is_weekend = df["weekday"].isin([0, 6])
    expected_wd = (~is_weekend & (df["holiday"] == 0)).astype(int)
    n_wd_bad = int((expected_wd != df["workingday"]).sum())
    report.add("workingday_rule", n_wd_bad == 0,
               f"{n_wd_bad} rows violate workingday = (not weekend) and (not holiday).")

    # Season coding: documented as 1=spring..4=winter, check how it maps onto the calendar.
    season_months = (pd.DataFrame({"season": df["season"], "month": parsed.dt.month})
                     .groupby("season")["month"].agg(lambda s: sorted(s.unique().tolist())))
    report.extras["season_to_months"] = {int(k): v for k, v in season_months.items()}
    report.add(
        "season_calendar_note", True,
        "Months observed per season code: "
        + "; ".join(f"{k}: {v}" for k, v in report.extras["season_to_months"].items())
        + ". Season changes at the astronomical equinox/solstice dates (~21st of Mar/Jun/Sep/Dec), "
          "so code 1 (README 'spring') mostly contains January-March dates.",
        severity="info",
    )

    # Weather extreme values worth flagging (not automatically errors).
    report.extras["zero_humidity_rows"] = int((df["hum"] == 0).sum())
    report.extras["zero_windspeed_rows"] = int((df["windspeed"] == 0).sum())
    zero_hum_dates = sorted(df.loc[df["hum"] == 0, "dteday"].unique().tolist())
    report.add("humidity_zero", report.extras["zero_humidity_rows"] == 0,
               f"{report.extras['zero_humidity_rows']} rows with humidity exactly 0 "
               f"(physically implausible; dates: {zero_hum_dates}). Kept but flagged.",
               severity="info")
    report.add("windspeed_zero", True,
               f"{report.extras['zero_windspeed_rows']} rows with wind speed exactly 0; plausible "
               "for calm conditions but may also encode missing readings. Kept as-is.",
               severity="info")


def validate_hourly(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport("hour.csv", len(df), df.shape[1])
    _common_checks(df, report, HOURLY_COLUMNS, config.EXPECTED_HOURLY_ROWS)

    parsed = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    full = pd.date_range(parsed.min(), parsed.max(), freq="h")
    missing_ts = full.difference(parsed)
    per_day = df.groupby("dteday").size()
    incomplete = per_day[per_day < 24]
    missing_by_day = (pd.Series(missing_ts.date).value_counts().sort_index())
    report.extras["hourly_gap_count"] = int(len(missing_ts))
    report.extras["days_with_missing_hours"] = int(len(incomplete))
    report.extras["largest_daily_gaps"] = {str(k): int(v) for k, v in
                                           missing_by_day.sort_values(ascending=False).head(10).items()}
    report.add(
        "hourly_sequence_gaps", True,
        f"{len(missing_ts)} of {len(full)} possible hourly timestamps are absent across "
        f"{len(incomplete)} dates. README describes the files as aggregated counts, so absent hours "
        "are most plausibly hours with no recorded rentals or system downtime, not dropped rows. "
        "They are documented, not imputed.", severity="info",
    )
    report.extras["hours_per_day_distribution"] = {int(k): int(v) for k, v in
                                                   per_day.value_counts().sort_index().items()}

    # Distributional outliers (flag only; extreme demand can be genuine).
    q1, q3 = df["cnt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    n_hi = int((df["cnt"] > upper).sum())
    # Hour-conditional robust z-score, since "high" depends strongly on the hour of day.
    g = df.groupby(["hr", "workingday"])["cnt"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    rz = 0.6745 * (df["cnt"] - med) / mad
    flagged = df.loc[rz.abs() > 3.5, ["dteday", "hr", "workingday", "weathersit", "cnt"]].copy()
    flagged["robust_z"] = rz[rz.abs() > 3.5].round(2)
    report.extras["cnt_iqr_upper_fence"] = float(upper)
    report.extras["cnt_iqr_outliers"] = n_hi
    report.extras["conditional_outliers"] = int(len(flagged))
    report.extras["conditional_outlier_examples"] = (
        flagged.reindex(flagged["robust_z"].abs().sort_values(ascending=False).index)
        .head(10).astype({"robust_z": float}).to_dict(orient="records"))
    report.add(
        "cnt_outliers", True,
        f"{n_hi} hourly counts exceed the global IQR fence ({upper:.0f}); {len(flagged)} rows have a "
        "|robust z| > 3.5 relative to the median for the same hour and working-day status. These are "
        "flagged for investigation, NOT removed.", severity="info",
    )
    return report


def validate_daily(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport("day.csv", len(df), df.shape[1])
    _common_checks(df, report, DAILY_COLUMNS, config.EXPECTED_DAILY_ROWS)
    dates = pd.to_datetime(df["dteday"])
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    report.add("daily_sequence_complete", len(full.difference(dates)) == 0,
               f"{len(full.difference(dates))} calendar dates missing from day.csv.")
    q1, q3 = df["cnt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df.loc[(df["cnt"] < lo) | (df["cnt"] > hi), ["dteday", "cnt", "weathersit"]]
    lowest = df.nsmallest(5, "cnt")[["dteday", "cnt", "weathersit", "holiday"]]
    report.extras["lowest_days"] = lowest.to_dict(orient="records")
    report.add("daily_cnt_outliers", True,
               f"{len(out)} days fall outside the IQR fences [{lo:.0f}, {hi:.0f}]. Five lowest days: "
               + ", ".join(f"{r.dteday} ({r.cnt})" for r in lowest.itertuples())
               + ". Retained as genuine (weather/event) observations.", severity="info")
    return report


def cross_check(hourly: pd.DataFrame, daily: pd.DataFrame) -> ValidationReport:
    """Aggregate hour.csv to days and compare with day.csv."""
    report = ValidationReport("hour.csv vs day.csv", len(daily), daily.shape[1])
    agg = hourly.groupby("dteday").agg(
        casual=("casual", "sum"), registered=("registered", "sum"), cnt=("cnt", "sum"),
        season=("season", "first"), yr=("yr", "first"), mnth=("mnth", "first"),
        holiday=("holiday", "first"), weekday=("weekday", "first"),
        workingday=("workingday", "first"),
    ).reset_index()
    report.add("same_dates", set(agg["dteday"]) == set(daily["dteday"]),
               f"{agg['dteday'].nunique()} dates in hour.csv vs {daily['dteday'].nunique()} in day.csv.")
    m = daily.merge(agg, on="dteday", suffixes=("_day", "_hour"))
    for col in ["casual", "registered", "cnt"]:
        diff = (m[f"{col}_day"] - m[f"{col}_hour"]).abs()
        report.add(f"daily_sum_{col}", int((diff > 0).sum()) == 0,
                   f"{int((diff > 0).sum())} dates where summed hourly `{col}` differs from day.csv "
                   f"(max abs diff {int(diff.max())}).")
    for col in ["season", "yr", "mnth", "holiday", "weekday", "workingday"]:
        n = int((m[f"{col}_day"] != m[f"{col}_hour"]).sum())
        report.add(f"calendar_{col}", n == 0, f"{n} dates where `{col}` disagrees.")
    # Within-day constancy of calendar attributes in hour.csv.
    const = hourly.groupby("dteday")[["season", "holiday", "weekday", "workingday"]].nunique()
    report.add("calendar_constant_within_day", bool((const == 1).all().all()),
               "Calendar attributes are constant within each date in hour.csv.")
    # Weather is NOT directly comparable: day.csv holds daily averages/situations, so only
    # report the correlation between daily mean hourly temp and day.csv temp.
    t = hourly.groupby("dteday")["temp"].mean().reindex(daily["dteday"]).to_numpy()
    r = float(np.corrcoef(t, daily["temp"])[0, 1])
    report.add("weather_temp_agreement", r > 0.95,
               f"Correlation between mean hourly temp and day.csv temp: r={r:.4f}. "
               "(Exact equality is not expected: the daily file uses its own daily aggregation.)",
               severity="info")
    return report


def run_all(hour_path: Path | str = config.HOUR_CSV, day_path: Path | str = config.DAY_CSV) -> dict:
    hourly = load_raw(hour_path)
    daily = load_raw(day_path)
    reports = [validate_hourly(hourly), validate_daily(daily), cross_check(hourly, daily)]
    return {"reports": [r.as_dict() for r in reports], "all_passed": all(r.passed for r in reports)}


def _to_markdown(results: dict) -> str:
    lines = [
        "# Data Quality Report",
        "",
        "Generated by `python -m src.data_validation` from the original UCI files in `data/raw/`.",
        "",
        f"Source: {config.UCI_URL}",
        "",
        f"**Overall result: {'PASS' if results['all_passed'] else 'FAIL'}** (all error-level checks; "
        "info-level items are observations, not defects).",
        "",
    ]
    for rep in results["reports"]:
        lines += [f"## {rep['dataset']}", "",
                  f"Rows: {rep['n_rows']:,} | Columns: {rep['n_cols']} | "
                  f"Result: {'PASS' if rep['passed'] else 'FAIL'}", "",
                  "| Check | Level | Result | Detail |", "|---|---|---|---|"]
        for c in rep["checks"]:
            status = "PASS" if c["passed"] else ("FLAG" if c["severity"] == "info" else "FAIL")
            detail = str(c["detail"]).replace("|", "\\|")
            lines.append(f"| `{c['name']}` | {c['severity']} | {status} | {detail} |")
        lines.append("")
        ex = rep["extras"]
        if "hours_per_day_distribution" in ex:
            lines += ["**Hours recorded per date** (hours: number of dates): "
                      + ", ".join(f"{k}: {v}" for k, v in ex["hours_per_day_distribution"].items()), "",
                      "**Dates with most missing hours:** "
                      + ", ".join(f"{k} ({v})" for k, v in ex["largest_daily_gaps"].items()), ""]
        if ex.get("conditional_outlier_examples"):
            lines += ["**Largest hour-conditional outliers (flagged, retained):**", "",
                      "| date | hr | workingday | weathersit | cnt | robust z |", "|---|---|---|---|---|---|"]
            for r in ex["conditional_outlier_examples"]:
                lines.append(f"| {r['dteday']} | {r['hr']} | {r['workingday']} | {r['weathersit']} | "
                             f"{r['cnt']} | {r['robust_z']} |")
            lines.append("")
    lines += [
        "## Decisions",
        "",
        "- Raw files are never modified; the cleaned/feature table is written separately to "
        "`data/processed/hour_features.csv`.",
        "- No rows are removed. Statistical outliers are flagged only, because extreme demand "
        "(or near-zero demand during storms) is real behaviour, not bad data.",
        "- Missing hourly timestamps are documented, not imputed. Model training uses only observed "
        "hours; daily commuting totals in the hypothesis test are computed over observed hours and "
        "re-checked using only days with complete commuting windows.",
        "- `casual` and `registered` are excluded from all predictors of `cnt` (they sum to the target).",
        "- `instant` is a row index and is never used as a feature.",
        "- Season labels follow the official README coding, with the calendar caveat documented above.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    config.ensure_dirs()
    results = run_all()
    (config.OUTPUTS_DIR / "data_quality.json").write_text(json.dumps(results, indent=2, default=str))
    (config.REPORTS_DIR / "data_quality_report.md").write_text(_to_markdown(results), encoding="utf-8")
    for rep in results["reports"]:
        failed = [c["name"] for c in rep["checks"] if not c["passed"] and c["severity"] == "error"]
        print(f"{rep['dataset']}: {'PASS' if rep['passed'] else 'FAIL'} {failed if failed else ''}")
    return results


if __name__ == "__main__":
    main()
