"""Feature engineering for the hourly bike-sharing data.

All transformations are deterministic functions of the raw columns (no target statistics), so the
same code can be applied to the training data, test data and single what-if scenarios in the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Official README decodings.
SEASON_LABELS = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
WEATHER_LABELS = {
    1: "clear / partly cloudy",
    2: "mist / cloudy",
    3: "light rain / light snow",
    4: "heavy rain / snow / fog",
}
WEATHER_SHORT = {1: "clear", 2: "mist", 3: "light rain/snow", 4: "heavy rain/snow"}
WEEKDAY_LABELS = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
                  5: "Friday", 6: "Saturday"}
MONTH_LABELS = {i: n for i, n in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# De-normalisation constants from the README.
TEMP_MAX_C = 41.0
ATEMP_MAX_C = 50.0
HUM_MAX = 100.0
WIND_MAX = 67.0

TARGET = "cnt"
# Columns that must never be used to predict cnt.
LEAKAGE_COLUMNS = ("casual", "registered", "cnt", "instant")

# Model inputs. Every feature is knowable before the hour occurs (calendar) or is a weather
# condition supplied by the user / a weather forecast.
MODEL_FEATURES = [
    "hr", "weekday", "mnth", "season", "yr", "holiday", "workingday", "is_weekend",
    "weathersit", "temp", "atemp", "hum", "windspeed",
    "hr_sin", "hr_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos",
    "is_commute_hour", "workday_commute",
]


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns required by the model. Works on raw-schema rows."""
    out = df.copy()
    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)
    out["hr_sin"], out["hr_cos"] = _cyclical(out["hr"], 24)
    out["weekday_sin"], out["weekday_cos"] = _cyclical(out["weekday"], 7)
    out["mnth_sin"], out["mnth_cos"] = _cyclical(out["mnth"] - 1, 12)
    out["is_commute_hour"] = out["hr"].isin(config.COMMUTE_HOURS).astype(int)
    out["workday_commute"] = (out["is_commute_hour"] & (out["workingday"] == 1)).astype(int)
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready copy of hour.csv with decoded and engineered features."""
    df = add_model_features(raw)
    df["date"] = pd.to_datetime(df["dteday"], format="%Y-%m-%d")
    df["datetime"] = df["date"] + pd.to_timedelta(df["hr"], unit="h")
    df["year"] = df["date"].dt.year
    df["season_label"] = df["season"].map(SEASON_LABELS)
    df["weather_label"] = df["weathersit"].map(WEATHER_LABELS)
    df["weather_short"] = df["weathersit"].map(WEATHER_SHORT)
    df["weekday_label"] = df["weekday"].map(WEEKDAY_LABELS)
    df["month_label"] = df["mnth"].map(MONTH_LABELS)
    df["day_type"] = np.where(df["workingday"] == 1, "working day", "non-working day")
    df["temp_c"] = df["temp"] * TEMP_MAX_C
    df["atemp_c"] = df["atemp"] * ATEMP_MAX_C
    df["hum_pct"] = df["hum"] * HUM_MAX
    df["windspeed_kmh"] = df["windspeed"] * WIND_MAX
    df["hour_period"] = pd.cut(
        df["hr"], bins=[-1, 5, 6, 8, 15, 18, 23],
        labels=["night (0-5)", "early (6)", "AM commute (7-8)", "midday (9-15)",
                "PM commute (16-18)", "evening (19-23)"],
    ).astype(str)
    return df


def assert_no_leakage(feature_names: list[str]) -> None:
    bad = sorted(set(feature_names) & set(LEAKAGE_COLUMNS))
    if bad:
        raise ValueError(f"Leakage columns present in model features: {bad}")


def scenario_frame(hr: int, season: int, workingday: int, weathersit: int, temp_c: float,
                   hum_pct: float, windspeed_kmh: float, yr: int = 1, mnth: int | None = None,
                   weekday: int | None = None, holiday: int = 0,
                   atemp_c: float | None = None) -> pd.DataFrame:
    """Build a single raw-schema row for a what-if prediction.

    Defaults: if month is unspecified we use the middle month of the README season; if the weekday
    is unspecified we use Wednesday for working days and Saturday for non-working days. The
    feeling temperature should be supplied by the caller (the model bundle stores a linear
    temp->atemp mapping fitted on training data); if omitted it falls back to ``temp_c``.
    """
    if mnth is None:
        mnth = {1: 2, 2: 5, 3: 8, 4: 11}[season]
    if weekday is None:
        weekday = 3 if workingday == 1 else 6
    if atemp_c is None:
        atemp_c = temp_c
    row = pd.DataFrame([{
        "season": season, "yr": yr, "mnth": mnth, "hr": hr, "holiday": holiday,
        "weekday": weekday, "workingday": workingday, "weathersit": weathersit,
        "temp": temp_c / TEMP_MAX_C, "atemp": atemp_c / ATEMP_MAX_C,
        "hum": hum_pct / HUM_MAX, "windspeed": windspeed_kmh / WIND_MAX,
    }])
    return add_model_features(row)


def main() -> pd.DataFrame:
    from src.data_validation import load_raw
    config.ensure_dirs()
    df = build_features(load_raw(config.HOUR_CSV))
    df.to_csv(config.PROCESSED_DIR / "hour_features.csv", index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns to data/processed/hour_features.csv")
    return df


if __name__ == "__main__":
    main()
