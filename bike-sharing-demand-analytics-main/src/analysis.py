"""Reusable, data-grounded summaries used by the dashboard and the natural-language layer.

Every function takes the feature table produced by ``src.features.build_features`` and returns
plain pandas objects computed from the data. No numbers are hard-coded.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.features import WEATHER_LABELS, WEATHER_SHORT, build_features

# Season naming schemes. "official" follows the README (1=spring ... 4=winter). In the data the
# codes switch on the astronomical dates (~21 Mar/Jun/Sep/Dec), so code 1 covers ~21 Dec-20 Mar.
# "calendar" relabels the same codes by the astronomical season they actually cover.
SEASON_SCHEMES = {
    "official": {1: "spring", 2: "summer", 3: "fall", 4: "winter"},
    "calendar": {1: "winter", 2: "spring", 3: "summer", 4: "fall"},
}
SEASON_DATE_RANGES = {1: "~21 Dec-20 Mar", 2: "~21 Mar-20 Jun", 3: "~21 Jun-22 Sep", 4: "~23 Sep-20 Dec"}


def season_code(name: str, scheme: str = "official") -> int:
    inv = {v: k for k, v in SEASON_SCHEMES[scheme].items()}
    return inv[name]


def season_name(code: int, scheme: str = "official") -> str:
    return SEASON_SCHEMES[scheme][int(code)]


@lru_cache(maxsize=1)
def load_features_cached(path: str = str(config.HOUR_CSV)) -> pd.DataFrame:
    from src.data_validation import load_raw
    return build_features(load_raw(path))


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("date")
            .agg(cnt=("cnt", "sum"), workingday=("workingday", "first"), season=("season", "first"),
                 yr=("yr", "first"), mnth=("mnth", "first"), weathersit_max=("weathersit", "max"),
                 temp_c=("temp_c", "mean"), hours=("hr", "count"))
            .reset_index())


def kpis(df: pd.DataFrame) -> dict:
    d = daily_table(df)
    by_year = d.groupby("yr")["cnt"].sum()
    prof = df.groupby("hr")["cnt"].mean()
    return {
        "total_rentals": int(df["cnt"].sum()),
        "n_hours": int(len(df)), "n_days": int(d["date"].nunique()),
        "mean_hourly": float(df["cnt"].mean()), "median_hourly": float(df["cnt"].median()),
        "mean_daily": float(d["cnt"].mean()),
        "max_hourly": int(df["cnt"].max()),
        "max_hourly_when": str(df.loc[df["cnt"].idxmax(), "datetime"]),
        "peak_hour_overall": int(prof.idxmax()),
        "total_2011": int(by_year.get(0, 0)), "total_2012": int(by_year.get(1, 0)),
        "yoy_growth_pct": float(100 * (by_year.get(1, 0) / by_year.get(0, 1) - 1)),
        "registered_share_pct": float(100 * df["registered"].sum() / df["cnt"].sum()),
    }


def hourly_profile(df: pd.DataFrame, by: str | None = "day_type") -> pd.DataFrame:
    if by is None:
        return df.groupby("hr")["cnt"].agg(["mean", "median", "std"]).reset_index()
    return df.groupby([by, "hr"])["cnt"].mean().unstack(0).reset_index()


def peak_hours(df: pd.DataFrame, workingday: int | None = None, top: int = 3,
               season: int | None = None) -> pd.DataFrame:
    sub = df
    if workingday is not None:
        sub = sub[sub["workingday"] == workingday]
    if season is not None:
        sub = sub[sub["season"] == season]
    prof = sub.groupby("hr")["cnt"].agg(mean="mean", median="median", p90=lambda s: s.quantile(0.9),
                                        n="count")
    return prof.sort_values("mean", ascending=False).head(top).reset_index()


def compare_day_types(df: pd.DataFrame) -> pd.DataFrame:
    d = daily_table(df)
    rows = []
    for wd, label in ((1, "working day"), (0, "non-working day")):
        h = df[df["workingday"] == wd]
        prof = h.groupby("hr")["cnt"].mean()
        rows.append({
            "day type": label, "days": int((d["workingday"] == wd).sum()),
            "mean rentals/day": float(d.loc[d["workingday"] == wd, "cnt"].mean()),
            "median rentals/day": float(d.loc[d["workingday"] == wd, "cnt"].median()),
            "mean rentals/hour": float(h["cnt"].mean()),
            "peak hour": int(prof.idxmax()), "peak-hour mean": float(prof.max()),
            "commute-hours mean/hour": float(h.loc[h["is_commute_hour"] == 1, "cnt"].mean()),
            "registered share %": float(100 * h["registered"].sum() / h["cnt"].sum()),
        })
    return pd.DataFrame(rows)


def weather_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean hourly demand by weather situation, raw and adjusted for time-of-day mix.

    The adjusted ratio compares each hour's count with the mean for the same
    (hour, working day, year, month) cell, so it is not driven by e.g. rain being more frequent
    at night or in winter."""
    cell = df.groupby(["hr", "workingday", "yr", "mnth"])["cnt"].transform("mean")
    t = df.assign(expected=cell)
    g = t.groupby("weathersit").agg(hours=("cnt", "size"), mean_hourly=("cnt", "mean"),
                                    median_hourly=("cnt", "median"), actual=("cnt", "sum"),
                                    expected=("expected", "sum"))
    g["adjusted_ratio"] = g["actual"] / g["expected"]
    g["adjusted_change_pct"] = 100 * (g["adjusted_ratio"] - 1)
    g = g.reset_index()
    g["weather"] = g["weathersit"].map(WEATHER_LABELS)
    return g[["weathersit", "weather", "hours", "mean_hourly", "median_hourly", "adjusted_change_pct"]]


def temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 5, 10, 15, 20, 25, 30, np.inf]
    labels = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"]
    t = df.assign(temp_band=pd.cut(df["temp_c"], bins=bins, labels=labels))
    return (t.groupby("temp_band", observed=True)["cnt"].agg(hours="size", mean_hourly="mean")
            .reset_index())


def season_summary(df: pd.DataFrame, scheme: str = "official") -> pd.DataFrame:
    d = daily_table(df)
    h = df.groupby("season")["cnt"].agg(mean_hourly="mean")
    dd = d.groupby("season")["cnt"].agg(days="size", mean_daily="mean", median_daily="median")
    out = h.join(dd).reset_index()
    out["season_name"] = out["season"].map(SEASON_SCHEMES[scheme])
    out["date_range"] = out["season"].map(SEASON_DATE_RANGES)
    return out[["season", "season_name", "date_range", "days", "mean_daily", "median_daily", "mean_hourly"]]


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = daily_table(df)
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    return d.groupby("month")["cnt"].agg(total="sum", mean_daily="mean").reset_index()


def monitor_hours(df: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Rank (day type, hour) slots by an operational-attention score.

    Score = 90th-percentile demand in that slot (how high demand gets on a busy day). Variability
    (std) and coefficient of variation are reported alongside so managers can see which slots are
    both busy and unpredictable."""
    g = df.groupby(["day_type", "hr"])["cnt"].agg(
        mean="mean", p90=lambda s: s.quantile(0.9), std="std", n="count").reset_index()
    g["cv"] = g["std"] / g["mean"]
    return g.sort_values("p90", ascending=False).head(top).reset_index(drop=True)


def scenario_defaults(df: pd.DataFrame, season: int, weathersit: int | None = None) -> dict:
    """Median weather conditions observed in a season (optionally for one weather situation)."""
    sub = df[df["season"] == season]
    if weathersit is not None and (sub["weathersit"] == weathersit).sum() >= 20:
        sub = sub[sub["weathersit"] == weathersit]
    return {"temp_c": float(sub["temp_c"].median()), "hum_pct": float(sub["hum_pct"].median()),
            "windspeed_kmh": float(sub["windspeed_kmh"].median())}


def typical_season_for_month(df: pd.DataFrame, month: int) -> int:
    return int(df.loc[df["mnth"] == month, "season"].mode().iloc[0])


def weather_short(code: int) -> str:
    return WEATHER_SHORT[int(code)]

