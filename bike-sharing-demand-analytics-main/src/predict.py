"""Load the persisted model bundle and produce explained scenario predictions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import WEATHER_LABELS, WEEKDAY_LABELS, scenario_frame
from src.train_model import HourWorkingdayBaseline

MODEL_PATH = config.MODELS_DIR / "demand_model.joblib"


class ModelNotFoundError(FileNotFoundError):
    pass


def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"Model artifact not found at {path}. Run `python -m src.train_model`.")
    if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "HourWorkingdayBaseline"):
        setattr(sys.modules["__main__"], "HourWorkingdayBaseline", HourWorkingdayBaseline)
    return joblib.load(path)


def _interval(pred: float, rq: dict) -> tuple[float, float]:
    for b in rq["bins"]:
        if b["lower_edge"] < pred <= b["upper_edge"]:
            return max(0.0, pred + b["q10_residual"]), max(0.0, pred + b["q90_residual"])
    return max(0.0, pred), max(0.0, pred)


def predict_scenario(bundle: dict, *, hr: int, season: int, workingday: int, weathersit: int,
                     temp_c: float, hum_pct: float, windspeed_kmh: float, yr: int = 1,
                     mnth: int | None = None, weekday: int | None = None, holiday: int = 0,
                     buffer_pct: float = 0.0) -> dict:
    a = bundle["atemp_from_temp"]
    atemp_c = a["slope"] * temp_c + a["intercept"]
    X = scenario_frame(hr=hr, season=season, workingday=workingday, weathersit=weathersit,
                       temp_c=temp_c, hum_pct=hum_pct, windspeed_kmh=windspeed_kmh, yr=yr,
                       mnth=mnth, weekday=weekday, holiday=holiday, atemp_c=atemp_c)
    pred = float(np.clip(bundle["model"].predict(X[bundle["features"]])[0], 0, None))
    base = float(np.clip(bundle["baseline"].predict(X)[0], 0, None))
    lo, hi = _interval(pred, bundle["residual_quantiles"])
    row = X.iloc[0]
    return {
        "estimate": pred,
        "estimate_rounded": int(round(pred)),
        "range80": (lo, hi),
        "baseline_estimate": base,
        "planning_buffer_pct": buffer_pct,
        "planning_target": int(math.ceil(pred * (1 + buffer_pct / 100.0))) if buffer_pct else None,
        "inputs": {
            "hour": f"{int(row.hr):02d}:00-{int(row.hr):02d}:59", "season_code": int(row.season),
            "month": int(row.mnth), "weekday": WEEKDAY_LABELS[int(row.weekday)],
            "working day": bool(row.workingday), "holiday": bool(row.holiday),
            "weather": WEATHER_LABELS[int(row.weathersit)], "temperature_c": round(temp_c, 1),
            "feels_like_c (derived)": round(atemp_c, 1), "humidity_pct": round(hum_pct, 0),
            "wind_kmh": round(windspeed_kmh, 1), "year level": 2011 + int(row.yr),
        },
    }


def day_profile(bundle: dict, **kwargs) -> pd.DataFrame:
    """Predicted demand for each hour of a day under fixed conditions."""
    kwargs.pop("hr", None)
    rows = [{"hr": h, "estimate": predict_scenario(bundle, hr=h, **kwargs)["estimate"]} for h in range(24)]
    return pd.DataFrame(rows)

