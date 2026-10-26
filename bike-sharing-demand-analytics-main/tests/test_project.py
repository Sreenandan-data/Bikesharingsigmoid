"""Automated tests for data validation, feature engineering, leakage prevention,
hypothesis testing, chronological splits, model predictions, NL query engine,
anomaly detection, and missing-data error handling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import analysis, anomaly, config, data_validation, features, hypothesis, predict, query_engine, train_model


@pytest.fixture(scope="session")
def raw_hourly() -> pd.DataFrame:
    return data_validation.load_raw(config.HOUR_CSV)


@pytest.fixture(scope="session")
def raw_daily() -> pd.DataFrame:
    return data_validation.load_raw(config.DAY_CSV)


@pytest.fixture(scope="session")
def feat_df(raw_hourly: pd.DataFrame) -> pd.DataFrame:
    return features.build_features(raw_hourly)


@pytest.fixture(scope="session")
def model_bundle() -> dict:
    return predict.load_bundle()


# ------------------------------------------------------------------
# 1. Data Validation & Missing-Data Handling
# ------------------------------------------------------------------
def test_uci_row_counts_and_columns(raw_hourly: pd.DataFrame, raw_daily: pd.DataFrame) -> None:
    assert len(raw_hourly) == config.EXPECTED_HOURLY_ROWS == 17_379
    assert len(raw_daily) == config.EXPECTED_DAILY_ROWS == 731
    assert list(raw_hourly.columns) == data_validation.HOURLY_COLUMNS
    assert list(raw_daily.columns) == data_validation.DAILY_COLUMNS


def test_validation_reports_pass(raw_hourly: pd.DataFrame, raw_daily: pd.DataFrame) -> None:
    rep_h = data_validation.validate_hourly(raw_hourly)
    rep_d = data_validation.validate_daily(raw_daily)
    rep_x = data_validation.cross_check(raw_hourly, raw_daily)
    assert rep_h.passed
    assert rep_d.passed
    assert rep_x.passed
    assert rep_h.extras["hourly_gap_count"] == 165
    assert rep_h.extras["days_with_missing_hours"] == 76


def test_missing_data_raises_clear_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(data_validation.DataNotFoundError, match="Source file not found"):
        data_validation.load_raw(missing_file)


def test_missing_model_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(predict.ModelNotFoundError, match="Model artifact not found"):
        predict.load_bundle(tmp_path / "no_model.joblib")


# ------------------------------------------------------------------
# 2. Feature Engineering & Leakage Prevention
# ------------------------------------------------------------------
def test_feature_generation_and_columns(feat_df: pd.DataFrame) -> None:
    expected_engineered = [
        "date", "datetime", "year", "season_label", "weather_label", "weather_short",
        "weekday_label", "month_label", "day_type", "temp_c", "atemp_c", "hum_pct",
        "windspeed_kmh", "is_weekend", "hr_sin", "hr_cos", "weekday_sin", "weekday_cos",
        "mnth_sin", "mnth_cos", "is_commute_hour", "workday_commute",
    ]
    for col in expected_engineered:
        assert col in feat_df.columns, f"Missing engineered column: {col}"
    assert len(feat_df) == 17_379
    assert (feat_df["cnt"] == feat_df["casual"] + feat_df["registered"]).all()


def test_leakage_prevention() -> None:
    assert features.TARGET == "cnt"
    for leak_col in ("casual", "registered", "cnt", "instant"):
        assert leak_col not in features.MODEL_FEATURES
    features.assert_no_leakage(features.MODEL_FEATURES)
    with pytest.raises(ValueError, match="Leakage columns present"):
        features.assert_no_leakage(features.MODEL_FEATURES + ["registered"])
    with pytest.raises(ValueError, match="Leakage columns present"):
        features.assert_no_leakage(features.MODEL_FEATURES + ["casual"])
    with pytest.raises(ValueError, match="Leakage columns present"):
        features.assert_no_leakage(features.MODEL_FEATURES + ["instant"])


# ------------------------------------------------------------------
# 3. Chronological Split, Training & Nonnegative Predictions
# ------------------------------------------------------------------
def test_chronological_split_and_rolling_folds(feat_df: pd.DataFrame) -> None:
    dev, test = train_model.chronological_split(feat_df)
    assert len(dev) + len(test) == len(feat_df)
    assert dev["datetime"].max() < test["datetime"].min()
    for fold_name, tr_idx, va_idx in train_model.rolling_origin_folds(dev):
        assert len(tr_idx) > 0 and len(va_idx) > 0, fold_name
        assert dev.loc[tr_idx, "datetime"].max() < dev.loc[va_idx, "datetime"].min()


def test_training_and_nonnegative_predictions(feat_df: pd.DataFrame, model_bundle: dict) -> None:
    sample = feat_df.iloc[:600].copy()
    m = train_model.fit("hgb_log", {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20}, sample, sample["cnt"])
    b = train_model.fit("baseline", {}, sample, sample["cnt"])
    preds = train_model.predict_counts(m, sample)
    base_preds = train_model.predict_counts(b, sample)
    assert preds.shape == (600,)
    assert base_preds.shape == (600,)
    assert np.all(preds >= 0.0)
    assert np.all(base_preds >= 0.0)

    # Test persisted model scenario prediction and nonnegative policy
    scen = predict.predict_scenario(
        model_bundle,
        hr=8,
        season=2,
        workingday=1,
        weathersit=1,
        temp_c=22.0,
        hum_pct=55.0,
        windspeed_kmh=12.0,
        buffer_pct=15.0,
    )
    assert scen["estimate"] >= 0.0
    assert scen["estimate_rounded"] >= 0
    assert scen["planning_target"] >= scen["estimate_rounded"]
    assert scen["range80"][0] <= scen["estimate"] <= scen["range80"][1]

    # Extreme cold night scenario must remain nonnegative
    extreme_low = predict.predict_scenario(
        model_bundle,
        hr=3,
        season=1,
        workingday=0,
        weathersit=4,
        temp_c=-10.0,
        hum_pct=100.0,
        windspeed_kmh=60.0,
    )
    assert extreme_low["estimate"] >= 0.0


# ------------------------------------------------------------------
# 4. Hypothesis Testing
# ------------------------------------------------------------------
def test_hypothesis_calculation(feat_df: pd.DataFrame) -> None:
    res = hypothesis.run(feat_df)
    p = res["primary"]
    assert p["n_working"] == 500
    assert p["n_nonworking"] == 231
    assert p["mean_working"] > p["mean_nonworking"]
    assert p["mean_difference"] == pytest.approx(p["mean_working"] - p["mean_nonworking"], rel=1e-6)
    assert p["welch_t"] > 15.0
    assert 0.0 < p["welch_p_one_sided"] < 1e-40
    assert res["reject_H0"] is True
    assert res["adjusted_ols_hac"]["workingday_multiplier"] > 1.5


# ------------------------------------------------------------------
# 5. Natural-Language Query Engine
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,expected_intent",
    [
        ("When is bike demand highest on working days?", "peak_hours"),
        ("Compare demand on working days and non-working days.", "compare_day_types"),
        ("How does weather affect rental demand?", "weather_effect"),
        ("What is the average demand in summer?", "season_average"),
        ("Estimate bike rentals at 8 AM in summer on a working day.", "predict"),
        ("Compare expected demand during clear weather and light rain.", "predict"),
        ("Which hours should the operations team monitor most closely?", "monitor_hours"),
        ("Was the working-day difference statistically significant?", "hypothesis"),
        ("How accurate is the demand model?", "model_performance"),
        ("Which days had unusual demand?", "anomalies"),
        ("How did demand grow from 2011 to 2012?", "trend"),
    ],
)
def test_nl_query_intents_and_answers(
    question: str,
    expected_intent: str,
    feat_df: pd.DataFrame,
    model_bundle: dict,
) -> None:
    ctx = query_engine.Context(
        df=feat_df,
        bundle=model_bundle,
        hypothesis=hypothesis.run(feat_df),
        model_metrics=None,
        anomalies=pd.read_csv(config.OUTPUTS_DIR / "anomalies.csv"),
        use_llm=False,
    )
    ans = query_engine.answer(question, ctx)
    assert ans.understood is True
    assert ans.intent == expected_intent
    assert len(ans.text) > 30


def test_nl_unrecognized_query_does_not_hallucinate(feat_df: pd.DataFrame, model_bundle: dict) -> None:
    ctx = query_engine.Context(df=feat_df, bundle=model_bundle, use_llm=False)
    ans = query_engine.answer("What is the capital of Mars and the price of tea?", ctx)
    assert ans.understood is False
    assert ans.intent == "unknown"
    assert "won't guess" in ans.text


# ------------------------------------------------------------------
# 6. Anomaly Detection & App Module Import
# ------------------------------------------------------------------
def test_anomaly_detection_flags_hurricane_sandy(feat_df: pd.DataFrame, raw_daily: pd.DataFrame) -> None:
    daily_anom = anomaly.detect_daily_anomalies(feat_df, raw_daily)
    flagged_dates = set(daily_anom.loc[daily_anom["is_anomaly"], "dteday"])
    assert "2012-10-29" in flagged_dates
    assert "2012-10-30" in flagged_dates


def test_streamlit_app_imports_cleanly() -> None:
    import app
    assert callable(app.main)


def test_streamlit_apptest_renders_and_answers_query() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 6
    # Submit a natural-language question through the chat_input widget
    at.chat_input[0].set_value("Estimate bike rentals at 8 AM in summer on a working day.").run()
    assert not at.exception
    markdown_texts = " ".join(m.value for m in at.markdown)
    assert "Expected demand:" in markdown_texts
