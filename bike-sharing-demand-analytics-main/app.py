"""Streamlit Operations Dashboard & Natural-Language Prototype for Capital Bikeshare Demand.

Launch::

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analysis, config
from src.data_validation import DataNotFoundError, load_raw
from src.features import WEATHER_LABELS, WEEKDAY_LABELS, build_features
from src.predict import ModelNotFoundError, day_profile, load_bundle, predict_scenario
from src.query_engine import SUPPORTED_EXAMPLES, Context, answer

st.set_page_config(
    page_title="Capital Bikeshare Demand & Operations Prototype",
    page_icon="🚲",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    return build_features(load_raw(config.HOUR_CSV))


@st.cache_resource(show_spinner=False)
def get_model_bundle() -> dict | None:
    try:
        return load_bundle()
    except ModelNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_json_output(filename: str) -> dict | None:
    p = config.OUTPUTS_DIR / filename
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def get_anomalies() -> pd.DataFrame | None:
    p = config.OUTPUTS_DIR / "anomalies.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def render_chart_spec(spec: dict | None) -> None:
    if not spec:
        return
    data = spec["data"]
    kind = spec.get("kind", "bar")
    x = spec["x"]
    y = spec["y"]
    title = spec.get("title", "")
    if kind == "line":
        fig = px.line(data, x=x, y=y, markers=True, title=title)
    else:
        fig = px.bar(data, x=x, y=y, barmode="group", title=title)
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("🚲 Capital Bikeshare Demand Analytics & Operations Prototype")
    st.caption(
        "UCI Bike Sharing Dataset (Washington, D.C., 2011–2012) • "
        "Data-grounded descriptive analytics, hypothesis testing, ML demand estimation, and natural-language Q&A"
    )

    try:
        df = get_features()
    except DataNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    bundle = get_model_bundle()
    hyp = get_json_output("hypothesis_results.json")
    model_metrics = get_json_output("model_metrics.json")
    anomalies_df = get_anomalies()

    # Sidebar controls
    with st.sidebar:
        st.header("Operational Settings")
        season_scheme_label = st.radio(
            "Season label convention",
            options=["Official UCI README (1=spring..4=winter)", "Astronomical calendar (1=winter..4=fall)"],
            index=0,
            help=(
                "In the UCI dataset, season codes switch at the equinoxes/solstices (~21st of Mar/Jun/Sep/Dec). "
                "Code 2 is labeled 'summer' in Readme.txt (~21 Mar–20 Jun), whereas Code 3 covers calendar summer (~21 Jun–22 Sep)."
            ),
        )
        season_scheme = "official" if season_scheme_label.startswith("Official") else "calendar"
        buffer_pct = float(
            st.slider(
                "Safety planning buffer (%)",
                min_value=0,
                max_value=40,
                value=15,
                step=5,
                help=(
                    "Optional operational planning buffer applied on top of the model's expected demand. "
                    "Clearly separated from the statistical prediction."
                ),
            )
        )
        st.divider()
        st.markdown(
            "**Important Operational Note**\n\n"
            "Model predictions represent **expected system-wide hourly bike rentals (`cnt`)**. "
            "Physical bikes required per station also depend on trip duration, station dock capacity, "
            "origin–destination flows, and rebalancing trucks, which are not observed in the UCI dataset."
        )

    ctx = Context(
        df=df,
        bundle=bundle,
        hypothesis=hyp,
        model_metrics=model_metrics,
        anomalies=anomalies_df,
        season_scheme=season_scheme,
        buffer_pct=buffer_pct,
    )

    # Top-level KPI banner
    k = analysis.kpis(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Rentals (2011–2012)", f"{k['total_rentals']:,}")
    c2.metric("Hourly Observations", f"{k['n_hours']:,}", f"{k['n_days']} days")
    c3.metric("Mean Daily Demand", f"{k['mean_daily']:,.0f} / day")
    c4.metric("Mean Hourly Demand", f"{k['mean_hourly']:.0f} / hr", f"Median {k['median_hourly']:.0f}/hr")
    c5.metric("YoY Growth (2011 → 2012)", f"+{k['yoy_growth_pct']:.1f}%")
    c6.metric("Registered User Share", f"{k['registered_share_pct']:.1f}%")

    tabs = st.tabs([
        "💬 Natural-Language Assistant",
        "📊 Demand Estimator (What-If)",
        "📈 Hourly, Seasonal & Day-Type Patterns",
        "🌦️ Weather & Temperature Impact",
        "🧪 Hypothesis Testing",
        "🤖 Model Performance & Diagnostics",
        "🚨 Anomaly & Event Detection",
    ])

    # ------------------------------------------------------------------
    # TAB 0: Natural-Language Assistant
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Ask Questions in Plain English")
        st.markdown(
            "Ask about peak hours, working vs. non-working days, seasonal averages, weather impacts, "
            "statistical test results, model accuracy, or specific hourly demand estimates. "
            "Every numerical answer is computed directly from the dataset or the trained model."
        )

        st.markdown("**Quick example questions (click to run):**")
        q_cols = st.columns(3)
        selected_example: str | None = None
        for idx, ex in enumerate(SUPPORTED_EXAMPLES[:9]):
            if q_cols[idx % 3].button(ex, key=f"ex_btn_{idx}", use_container_width=True):
                selected_example = ex

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "text": (
                        "Hello! Ask me a question about Capital Bikeshare demand or click one of the "
                        "example prompts above."
                    ),
                    "intent": "welcome",
                    "table": None,
                    "chart": None,
                }
            ]

        user_prompt = st.chat_input("e.g., Estimate bike rentals at 8 AM in summer on a working day")
        active_query = selected_example or user_prompt

        if active_query:
            st.session_state.chat_history.append({"role": "user", "text": active_query})
            ans = answer(active_query, ctx)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "text": ans.text,
                    "intent": ans.intent,
                    "table": ans.table,
                    "chart": ans.chart,
                }
            )

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                if item.get("intent") and item["role"] == "assistant" and item["intent"] != "welcome":
                    st.caption(f"Detected intent: `{item['intent']}`")
                st.markdown(item["text"])
                if item.get("table") is not None:
                    st.dataframe(item["table"], use_container_width=True, hide_index=True)
                if item.get("chart") is not None:
                    render_chart_spec(item["chart"])

    # ------------------------------------------------------------------
    # TAB 1: Demand Estimator (What-If)
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Hourly Rental Demand Estimator")
        if bundle is None:
            st.warning("Trained model not found. Run `python -m src.train_model` first.")
        else:
            col_in, col_out = st.columns([1, 1.35])
            with col_in:
                st.markdown("#### Scenario Inputs")
                season_map = analysis.SEASON_SCHEMES[season_scheme]
                season_options = {
                    f"{code}: {name.title()} ({analysis.SEASON_DATE_RANGES[code]})": code
                    for code, name in season_map.items()
                }
                chosen_season_label = st.selectbox("Season", list(season_options.keys()), index=1)
                season_code = season_options[chosen_season_label]
                hr_val = st.slider("Hour of day (24h)", 0, 23, 8)
                day_mode = st.radio(
                    "Day classification",
                    ["Working day (Mon–Fri, non-holiday)", "Weekend (Sat–Sun)", "Public holiday"],
                    index=0,
                )
                workingday_val = 1 if day_mode.startswith("Working") else 0
                holiday_val = 1 if day_mode == "Public holiday" else 0
                weekday_default = 3 if workingday_val == 1 else (1 if holiday_val == 1 else 6)
                weekday_val = st.selectbox(
                    "Day of week",
                    options=list(WEEKDAY_LABELS.keys()),
                    format_func=lambda k: WEEKDAY_LABELS[k],
                    index=weekday_default,
                )
                weather_val = st.selectbox(
                    "Weather situation",
                    options=list(WEATHER_LABELS.keys()),
                    format_func=lambda k: f"{k}: {WEATHER_LABELS[k]}",
                    index=0,
                )
                defs = analysis.scenario_defaults(df, season_code, weather_val)
                temp_val = st.slider("Temperature (°C)", -5.0, 39.0, round(defs["temp_c"], 1), 0.5)
                hum_val = st.slider("Relative humidity (%)", 10.0, 100.0, round(defs["hum_pct"], 0), 1.0)
                wind_val = st.slider("Wind speed (km/h)", 0.0, 55.0, round(defs["windspeed_kmh"], 1), 0.5)
                yr_label = st.selectbox("Demand regime (year level)", ["2012 (Year 2 regime)", "2011 (Year 1 regime)"], index=0)
                yr_val = 1 if yr_label.startswith("2012") else 0

            with col_out:
                res = predict_scenario(
                    bundle,
                    hr=hr_val,
                    season=season_code,
                    workingday=workingday_val,
                    weathersit=weather_val,
                    temp_c=temp_val,
                    hum_pct=hum_val,
                    windspeed_kmh=wind_val,
                    yr=yr_val,
                    weekday=weekday_val,
                    holiday=holiday_val,
                    buffer_pct=buffer_pct,
                )
                st.markdown("#### Predicted Demand & Operational Interpretation")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Rentals", f"{res['estimate_rounded']:,} / hr")
                m2.metric("Indicative 80% Range", f"{res['range80'][0]:.0f}–{res['range80'][1]:.0f}")
                m3.metric("Baseline Estimate", f"{res['baseline_estimate']:.0f} / hr")
                m4.metric(
                    f"Planning Target (+{buffer_pct:.0f}%)",
                    f"{res['planning_target']:,} / hr" if res["planning_target"] else "No buffer",
                    help="Operational safety buffer assumption (not a model output).",
                )
                if weather_val == 4:
                    st.warning(
                        "Category 4 (Heavy rain / snow / ice pellets) appears in only 3 hours in the entire "
                        "dataset; treat predictions for this category with caution."
                    )

                prof24 = day_profile(
                    bundle,
                    season=season_code,
                    workingday=workingday_val,
                    weathersit=weather_val,
                    temp_c=temp_val,
                    hum_pct=hum_val,
                    windspeed_kmh=wind_val,
                    yr=yr_val,
                    weekday=weekday_val,
                    holiday=holiday_val,
                )
                fig24 = go.Figure()
                fig24.add_trace(
                    go.Bar(
                        x=prof24["hr"],
                        y=prof24["estimate"].round(0),
                        marker_color=["#d95f02" if h == hr_val else "#2b5c8f" for h in prof24["hr"]],
                        name="Expected rentals",
                    )
                )
                fig24.update_layout(
                    title=f"Full-Day Expected Hourly Profile (Selected hour {hr_val:02d}:00 highlighted)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Expected Rentals / Hour",
                    height=340,
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig24, use_container_width=True)

                with st.expander("View exact model inputs & assumptions", expanded=False):
                    st.json(res["inputs"])

    # ------------------------------------------------------------------
    # TAB 2: Hourly, Seasonal & Day-Type Patterns
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Hourly, Working-Day & Seasonal Demand Patterns")
        col_a, col_b = st.columns(2)
        with col_a:
            prof_day = analysis.hourly_profile(df, by="day_type")
            fig_hd = px.line(
                prof_day,
                x="hr",
                y=["working day", "non-working day"],
                markers=True,
                title="Mean Hourly Rentals: Working Day vs. Non-Working Day",
                labels={"hr": "Hour of Day", "value": "Mean Rentals / Hour", "variable": "Day Type"},
            )
            st.plotly_chart(fig_hd, use_container_width=True)
        with col_b:
            s_tab = analysis.season_summary(df, season_scheme)
            fig_s = px.bar(
                s_tab,
                x="season_name",
                y="mean_daily",
                text_auto=".0f",
                color="season_name",
                title=f"Mean Daily Rentals by Season ({season_scheme.title()} Scheme)",
                labels={"season_name": "Season", "mean_daily": "Mean Rentals / Day"},
            )
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("#### Working-Day vs. Non-Working-Day Summary Table")
        st.dataframe(analysis.compare_day_types(df).round(1), use_container_width=True, hide_index=True)

        st.markdown("#### Hours Requiring Closest Operational Monitoring (Ranked by 90th-Percentile Demand)")
        st.dataframe(analysis.monitor_hours(df, top=8).round(2), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 3: Weather & Temperature Impact
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Weather Situation & Temperature Analysis")
        w_tab = analysis.weather_summary(df)
        t_tab = analysis.temperature_summary(df)
        cw1, cw2 = st.columns(2)
        with cw1:
            fig_w = px.bar(
                w_tab,
                x="weather",
                y="mean_hourly",
                text_auto=".0f",
                color="weather",
                title="Mean Hourly Rentals by Weather Category",
            )
            st.plotly_chart(fig_w, use_container_width=True)
        with cw2:
            fig_t = px.bar(
                t_tab.astype({"temp_band": str}),
                x="temp_band",
                y="mean_hourly",
                text_auto=".0f",
                title="Mean Hourly Rentals by Temperature Band (°C)",
            )
            st.plotly_chart(fig_t, use_container_width=True)

        st.markdown(
            "**Weather Summary (Raw vs. Time-of-Day/Month/Year Adjusted):** "
            "`adjusted_change_pct` compares each hour against the mean of the same `(hr, workingday, yr, mnth)` cell."
        )
        st.dataframe(w_tab.round(1), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 4: Hypothesis Testing
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Statistical Hypothesis Testing: Commuting-Window Demand")
        if hyp is None:
            st.warning("Hypothesis results not found. Run `python -m src.hypothesis`.")
        else:
            p = hyp["primary"]
            adj = hyp["adjusted_ols_hac"]
            st.markdown(
                f"- **Research Question:** {hyp['research_question']}\n"
                f"- **$H_0$:** {hyp['H0']}\n"
                f"- **$H_1$:** {hyp['H1']}\n"
                f"- **Predefined Commuting Windows:** `{hyp['commute_windows']['description']}` "
                f"(Significance level $\\alpha = {hyp['alpha']}$)"
            )
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Working Days Mean (n=500)", f"{p['mean_working']:,.1f} / day", f"Median {p['median_working']:,.0f}")
            h2.metric("Non-Working Mean (n=231)", f"{p['mean_nonworking']:,.1f} / day", f"Median {p['median_nonworking']:,.0f}")
            h3.metric(
                "Mean Difference",
                f"+{p['mean_difference']:,.1f} / day",
                f"95% CI [{p['ci95_mean_difference_welch'][0]:.0f}, {p['ci95_mean_difference_welch'][1]:.0f}]",
            )
            h4.metric(
                "Welch t-statistic / p-value",
                f"t = {p['welch_t']:.2f}",
                f"p = {p['welch_p_one_sided']:.2e} (Hedges' g = {p['hedges_g']:.2f})",
            )
            st.success(
                f"**Conclusion:** Reject $H_0$ ($p = {p['welch_p_one_sided']:.2e} \\ll 0.05$). "
                f"After controlling for temperature, humidity, wind, weather category, month, and year with "
                f"Newey-West HAC standard errors (7 lags), working days still carry **{adj['workingday_multiplier']:.2f}x** "
                f"(95% CI {adj['workingday_multiplier_ci95'][0]:.2f}–{adj['workingday_multiplier_ci95'][1]:.2f}x) "
                f"as many commuting-window rentals as non-working days. Because this is observational data, "
                f"this finding reflects strong operational association rather than an experimental intervention."
            )
            fig_path = config.FIGURES_DIR / "hypothesis_commute.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Commuting-Window Demand Distribution, Group Means & Hourly Profile")

    # ------------------------------------------------------------------
    # TAB 5: Model Performance & Diagnostics
    # ------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Demand-Estimation Model Evaluation (Chronological Holdout: Jul–Dec 2012)")
        if model_metrics is None:
            st.warning("Model evaluation metrics not found. Run `python -m src.train_model`.")
        else:
            tm = model_metrics["test_metrics_hourly"]
            imp = model_metrics["test_improvement_vs_baseline"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Selected Model MAE", f"{tm['model']['MAE']:.2f} / hr", f"-{imp['MAE_reduction_pct']:.1f}% vs baseline")
            mc2.metric("Selected Model RMSE", f"{tm['model']['RMSE']:.2f} / hr", f"-{imp['RMSE_reduction_pct']:.1f}% vs baseline")
            mc3.metric("Selected Model R²", f"{tm['model']['R2']:.3f}", f"Baseline R² {tm['baseline']['R2']:.3f}")
            mc4.metric("80% Interval Coverage", f"{100 * model_metrics['interval_80pct_empirical_test_coverage']:.1f}%", "Target: 80.0%")

            cv_rows = [
                {"Model Family": k, "Best Params": str(v["params"]), "CV Mean RMSE": round(v["mean_rmse"], 2), "CV Mean MAE": round(v["mean_mae"], 2)}
                for k, v in model_metrics["cv_best_by_family"].items()
            ]
            st.markdown("#### Expanding-Window Cross-Validation Summary (Development Period: Jan 2011 – Jun 2012)")
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

            for fname, cap in [
                ("model_actual_vs_predicted.png", "Actual vs. Predicted Hourly Rentals (Test Set)"),
                ("model_residuals.png", "Residual Diagnostics & Hourly Error Profile"),
                ("model_time_series.png", "Chronological Test Window Comparison"),
                ("model_permutation_importance.png", "Permutation Feature Importance on Test Set"),
            ]:
                fpath = config.FIGURES_DIR / fname
                if fpath.exists():
                    st.image(str(fpath), caption=cap)

    # ------------------------------------------------------------------
    # TAB 6: Anomaly & Event Detection
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Conditional Anomaly & Event Detection (Stretch Goal)")
        st.markdown(
            "Candidate anomalies are days where observed rentals deviate sharply ($|z_{\\text{robust}}| \\ge 3.5$ "
            "or $\\le -65\\%$ drop on high-demand days) from cross-fitted expected demand conditioned on "
            "calendar, weather, and year."
        )
        anom_fig = config.FIGURES_DIR / "anomaly_detection.png"
        if anom_fig.exists():
            st.image(str(anom_fig), caption="Observed vs. Conditional Expected Daily Demand & Flagged Anomalies")
        if anomalies_df is not None:
            st.dataframe(anomalies_df.round(2), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

