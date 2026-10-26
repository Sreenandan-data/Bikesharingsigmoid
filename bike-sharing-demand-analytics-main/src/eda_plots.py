"""Generate exploratory and data-quality figures under ``outputs/figures/``."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import analysis, config
from src.features import build_features


def generate_all(df: pd.DataFrame | None = None) -> list[str]:
    if df is None:
        from src.data_validation import load_raw
        df = build_features(load_raw(config.HOUR_CSV))
    config.ensure_dirs()
    fd = config.FIGURES_DIR
    saved: list[str] = []

    # 1. Hourly demand profile by working day vs non-working day + seasonal profiles
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    prof = df.groupby(["day_type", "hr"])["cnt"].mean().unstack(0)
    for col in prof.columns:
        axes[0].plot(prof.index, prof[col], marker="o", lw=2, label=col)
    axes[0].set_title("Mean Hourly Demand: Working vs Non-Working Days")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Mean Rentals / Hour")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()

    sprof = df.groupby(["season_label", "hr"])["cnt"].mean().unstack(0)
    for col in sprof.columns:
        axes[1].plot(sprof.index, sprof[col], marker="o", lw=1.8, label=col)
    axes[1].set_title("Mean Hourly Demand by Season (Official README Labels)")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Mean Rentals / Hour")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fd / "eda_hourly_and_seasonal.png", dpi=130)
    plt.close(fig)
    saved.append("eda_hourly_and_seasonal.png")

    # 2. Weather situation and temperature relationship
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    w = analysis.weather_summary(df)
    axes[0].bar(w["weather"], w["mean_hourly"], color=["#2b5c8f", "#5c88b5", "#d9822b", "#b8332a"][: len(w)])
    axes[0].set_title("Mean Hourly Rentals by Weather Category")
    axes[0].set_ylabel("Mean Rentals / Hour")
    axes[0].tick_params(axis="x", rotation=18)

    t = analysis.temperature_summary(df)
    axes[1].bar(t["temp_band"].astype(str), t["mean_hourly"], color="#2e8b57")
    axes[1].set_title("Mean Hourly Rentals by Temperature Band (°C)")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Mean Rentals / Hour")
    fig.tight_layout()
    fig.savefig(fd / "eda_weather_and_temperature.png", dpi=130)
    plt.close(fig)
    saved.append("eda_weather_and_temperature.png")

    # 3. Data quality & completeness overview (hours observed per day across 2011-2012)
    per_day = df.groupby("date").agg(hours=("hr", "count"), cnt=("cnt", "sum")).reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    axes[0].plot(per_day["date"], per_day["cnt"], lw=1.0, color="#1f77b4")
    axes[0].set_title("Daily Total Bike Rentals (2011-2012)")
    axes[0].set_ylabel("Rentals / Day")
    axes[1].plot(per_day["date"], 24 - per_day["hours"], lw=1.0, color="#d62728")
    axes[1].set_title("Missing Hourly Timestamps per Date (165 Total Absent Hours Across 76 Dates)")
    axes[1].set_ylabel("Missing Hours")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(fd / "data_quality_gaps_and_daily_trend.png", dpi=130)
    plt.close(fig)
    saved.append("data_quality_gaps_and_daily_trend.png")

    return saved


if __name__ == "__main__":
    out = generate_all()
    print("Generated figures:", ", ".join(out))

