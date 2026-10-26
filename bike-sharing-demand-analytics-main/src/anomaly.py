"""Conditional anomaly and event detection on the UCI Bike Sharing Dataset.

Instead of flagging high raw rental counts (which simply reflect warm summer commuting peaks), this
module fits an 8-fold contiguous block cross-fitted HistGradientBoostingRegressor for 24-hour daily
demand conditioned on calendar, weather, and year features (using ``day.csv`` alongside hourly
completeness counts from ``hour.csv``). This ensures that days when the system was shut down for
many hours (such as Hurricane Sandy on 2012-10-29, when only 1 hour was logged) are compared against
full-day expected demand rather than only the 1 logged hour.

We compute both the level residual and log-ratio residual, standardized via the Median Absolute
Deviation (MAD):

    z_robust = 0.6745 * (residual - median(residual)) / MAD(residual)

Days with |z_robust| >= 3.0 or a shortfall >= 60% (when expected >= 1,500 rentals/day) are flagged
as candidate anomalies. Known historical Washington, D.C. events are annotated as temporal
associations (not proven causal links).

Run: ``python -m src.anomaly``
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src import config
from src.data_validation import load_raw
from src.features import TEMP_MAX_C, build_features

ROBUST_Z_THRESHOLD = 3.0

DAILY_MODEL_FEATURES = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]

# Documented historical Washington, D.C. events during 2011-2012 (temporal associations only).
KNOWN_DC_EVENTS = {
    "2012-10-29": "Hurricane Sandy landfall; Capital Bikeshare suspended service Oct 29-30, 2012 (Fanaee-T & Gama 2013)",
    "2012-10-30": "Hurricane Sandy aftermath; partial afternoon service resumption (Oct 30, 2012)",
    "2011-08-27": "Hurricane Irene approach & evening service suspension in D.C. (Aug 27-28, 2011)",
    "2011-08-28": "Hurricane Irene tropical storm winds & rain in D.C. (Aug 28, 2011)",
    "2011-01-26": "January 2011 Mid-Atlantic winter storm / heavy snow & sleet in D.C.",
    "2011-01-27": "Post-storm snow/ice recovery following Jan 26, 2011 snowstorm (16 missing hours)",
    "2011-01-18": "Freezing rain & ice storm in Washington, D.C. (Jan 18, 2011; 12 missing hours)",
    "2012-12-24": "Christmas Eve federal holiday period (Dec 24, 2012)",
    "2012-12-25": "Christmas Day (Dec 25, 2012)",
    "2012-12-26": "Post-Christmas winter storm & holiday week (Dec 26, 2012)",
    "2012-11-22": "U.S. Thanksgiving Day (Nov 22, 2012)",
    "2012-11-23": "Day after Thanksgiving / Black Friday (coded workingday=1 in dataset)",
    "2011-11-25": "Day after Thanksgiving / Black Friday (coded workingday=1 in dataset)",
    "2012-03-17": "St. Patrick's Day & warm National Cherry Blossom Festival weekend (Mar 17, 2012)",
    "2012-03-23": "Peak National Cherry Blossom Festival warm Friday in D.C. (Mar 23, 2012)",
    "2012-07-04": "U.S. Independence Day celebrations on the National Mall (Jul 4, 2012)",
}


def detect_daily_anomalies(
    hourly_df: pd.DataFrame,
    daily_raw: pd.DataFrame | None = None,
    z_threshold: float = ROBUST_Z_THRESHOLD,
) -> pd.DataFrame:
    """Return daily table with cross-fitted expected demand, residuals, robust z-scores, and flags."""
    if daily_raw is None:
        daily_raw = load_raw(config.DAY_CSV)
    daily = daily_raw.sort_values("dteday").reset_index(drop=True).copy()
    hours_obs = hourly_df.groupby("dteday")["hr"].count()
    weathersit_max = hourly_df.groupby("dteday")["weathersit"].max()

    kf = KFold(n_splits=8, shuffle=False)
    expected = np.zeros(len(daily), dtype=float)
    for tr, va in kf.split(daily):
        base = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            max_iter=300,
            early_stopping=False,
            random_state=config.RANDOM_STATE,
        )
        reg = TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
        reg.fit(daily.iloc[tr][DAILY_MODEL_FEATURES], daily.iloc[tr]["cnt"])
        expected[va] = np.clip(reg.predict(daily.iloc[va][DAILY_MODEL_FEATURES]), 0, None)

    out = pd.DataFrame({
        "dteday": daily["dteday"],
        "date": pd.to_datetime(daily["dteday"]),
        "actual": daily["cnt"].astype(int),
        "expected": expected,
        "hours_observed": daily["dteday"].map(hours_obs).fillna(0).astype(int),
        "workingday": daily["workingday"].astype(int),
        "holiday": daily["holiday"].astype(int),
        "weathersit_max": daily["dteday"].map(weathersit_max).fillna(daily["weathersit"]).astype(int),
        "temp_c": daily["temp"] * TEMP_MAX_C,
    })
    out["residual"] = out["actual"] - out["expected"]
    out["pct_diff"] = 100.0 * out["residual"] / np.maximum(out["expected"], 1.0)
    med = float(out["residual"].median())
    mad = float((out["residual"] - med).abs().median())
    out["robust_z"] = 0.6745 * (out["residual"] - med) / mad
    out["is_anomaly"] = (out["robust_z"].abs() >= z_threshold) | (
        (out["expected"] >= 1500) & (out["pct_diff"] <= -60.0)
    )
    out["direction"] = np.where(out["residual"] < 0, "low", "high")
    out["associated_event"] = out["dteday"].map(KNOWN_DC_EVENTS).fillna("")
    return out


def make_plot(daily: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True)
    axes[0].plot(daily["date"], daily["actual"], label="Observed daily rentals", lw=1.1, alpha=0.85)
    axes[0].plot(daily["date"], daily["expected"], label="Conditional expected (weather + calendar)", lw=1.1, alpha=0.85)
    anom = daily[daily["is_anomaly"]]
    axes[0].scatter(anom["date"], anom["actual"], color="crimson", s=28, zorder=5, label="Flagged candidate anomaly")
    axes[0].set_ylabel("Daily rentals")
    axes[0].set_title("Daily Bike Rentals vs. Conditional Expectation & Candidate Anomalies (2011-2012)")
    axes[0].legend()

    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhline(ROBUST_Z_THRESHOLD, color="crimson", ls="--", lw=0.9, label=f"+/- {ROBUST_Z_THRESHOLD} robust z")
    axes[1].axhline(-ROBUST_Z_THRESHOLD, color="crimson", ls="--", lw=0.9)
    axes[1].plot(daily["date"], daily["robust_z"], lw=1.0, color="#2b5c8f")
    axes[1].scatter(anom["date"], anom["robust_z"], color="crimson", s=28, zorder=5)
    axes[1].set_ylabel("Robust z-score of residual")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run() -> pd.DataFrame:
    config.ensure_dirs()
    hourly_df = build_features(load_raw(config.HOUR_CSV))
    daily_raw = load_raw(config.DAY_CSV)
    daily = detect_daily_anomalies(hourly_df, daily_raw)
    make_plot(daily, config.FIGURES_DIR / "anomaly_detection.png")
    flagged = (
        daily[daily["is_anomaly"]]
        .sort_values("robust_z", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    cols = [
        "dteday", "date", "actual", "expected", "residual", "pct_diff",
        "robust_z", "direction", "hours_observed", "workingday",
        "holiday", "weathersit_max", "temp_c", "associated_event",
    ]
    flagged[cols].to_csv(config.OUTPUTS_DIR / "anomalies.csv", index=False)
    summary = {
        "threshold_robust_z": ROBUST_Z_THRESHOLD,
        "secondary_rule": "pct_diff <= -60% when expected >= 1500",
        "n_days_total": int(len(daily)),
        "n_flagged": int(len(flagged)),
        "n_low": int((flagged["direction"] == "low").sum()),
        "n_high": int((flagged["direction"] == "high").sum()),
        "top_candidates": flagged[cols].head(20).astype({"date": str}).to_dict(orient="records"),
    }
    (config.OUTPUTS_DIR / "anomalies.json").write_text(json.dumps(summary, indent=2))
    print(f"Flagged {len(flagged)} candidate anomaly days out of {len(daily)} days.")
    return flagged


if __name__ == "__main__":
    run()

