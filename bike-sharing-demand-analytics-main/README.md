# Capital Bikeshare Demand Analytics & Natural-Language Operations Prototype

End-to-end data quality validation, feature engineering, statistical hypothesis testing, time-aware hourly demand estimation, conditional anomaly detection, and an interactive Streamlit natural-language operations dashboard built on the official **UCI Bike Sharing Dataset** (Washington, D.C., Capital Bikeshare, 2011–2012).

---

## 1. Problem Statement & Assignment Scope

Operations managers for a city bike-sharing system need to understand when and why rental demand surges or drops, validate behavioral hypotheses with rigorous statistics, estimate expected hourly bike demand under varying calendar and weather scenarios, and explore the data through plain-English questions without writing code.

An initial workspace audit (`reports/initial_audit.md`) examined the pre-existing R report (`time-series-master/time-series-master/time-series-analysis.html` and `.Rmd`, authored by Rebecca Hadi, 2018). That reference report performed an introductory daily ARIMA walkthrough on `day.csv` using pre-split `forecast::tsclean()` outlier removal and STL smoothing, but did not analyze `hour.csv`, did not test commuting-window hypotheses, did not build an exogenous hourly demand estimator, and did not provide an interactive operations tool. The original HTML and Rmd files are preserved untouched, and this repository implements a complete, leak-free Python solution from scratch.

---

## 2. Dataset Source, Schema & Record Counts

- **Official Source**: [UCI Machine Learning Repository — Bike Sharing Dataset (ID 275)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **System & Period**: Capital Bikeshare, Washington, D.C., USA — January 1, 2011 to December 31, 2012 (2 full calendar years).
- **Files & Verified Record Counts**:
  - `data/raw/hour.csv`: **17,379 hourly observations** across 17 columns.
  - `data/raw/day.csv`: **731 daily observations** across 16 columns (all columns except `hr`).

| Column | Type | Description & Official Domain |
|---|---|---|
| `instant` | `int` | Sequential record index (`1..17379` in `hour.csv`; `1..731` in `day.csv`). **Excluded from modeling.** |
| `dteday` | `str` | Calendar date (`YYYY-MM-DD`, `2011-01-01` to `2012-12-31`). |
| `season` | `int` | `1`: spring, `2`: summer, `3`: fall, `4`: winter (see season-coding note below). |
| `yr` | `int` | Year index (`0`: 2011, `1`: 2012). |
| `mnth` | `int` | Month (`1` to `12`). |
| `hr` | `int` | Hour of day (`0` to `23`, present in `hour.csv` only). |
| `holiday` | `int` | `1` if D.C. public holiday (21 dates), `0` otherwise. |
| `weekday` | `int` | Day of week (`0` = Sunday, `1` = Monday, ..., `6` = Saturday). |
| `workingday` | `int` | `1` if neither weekend nor holiday (500 days), `0` otherwise (231 days). |
| `weathersit` | `int` | `1`: Clear / partly cloudy (11,413 hrs), `2`: Mist / cloudy (4,544 hrs), `3`: Light snow / light rain (1,419 hrs), `4`: Heavy rain / ice pellets / snow + fog (3 hrs). |
| `temp` | `float` | Normalized temperature in Celsius, divided by $41^\circ\text{C}$ (`[0.02, 1.00]`). |
| `atemp` | `float` | Normalized feeling temperature in Celsius, divided by $50^\circ\text{C}$ (`[0.00, 1.00]`). |
| `hum` | `float` | Normalized relative humidity, divided by $100\%$ (`[0.00, 1.00]`). |
| `windspeed` | `float` | Normalized wind speed, divided by $67\text{ km/h}$ (`[0.00, 0.8507]`). |
| `casual` | `int` | Casual (non-registered) bike rentals. **Excluded from predictors to prevent target leakage.** |
| `registered` | `int` | Registered subscriber bike rentals. **Excluded from predictors to prevent target leakage.** |
| `cnt` | `int` | **Target variable**: Total bike rentals (`cnt == casual + registered`). |

---

## 3. Data Quality Checks & Feature Engineering

Implemented in `src/data_validation.py` (`reports/data_quality_report.md`, `outputs/data_quality.json`) and `src/features.py` (`data/processed/hour_features.csv`):

1. **Zero Schema or Arithmetic Defects**:
   - 0 missing values (`NaN`), 0 duplicate rows, 0 duplicate `(dteday, hr)` keys, and strictly monotonic timestamps.
   - `cnt == casual + registered` holds on all 17,379 hourly rows and all 731 daily rows.
   - Daily sums of `casual`, `registered`, and `cnt` aggregated from `hour.csv` match `day.csv` with **0 discrepancies across all 731 days**.
   - Calendar rules (`yr`, `mnth`, `weekday`, and `workingday == (~weekend & ~holiday)`) hold with **0 violations**.
2. **Hourly Sequence Gaps (165 Absent Hours)**:
   - Out of $731 \times 24 = 17,544$ possible hourly timestamps, **165 hours are absent across 76 dates** (`655` days have 24 hours; `62` have 23 hours; `14` have $\le 22$ hours).
   - The largest gaps occur during major storms/shutdowns (`2012-10-29`: 23 missing hours during Hurricane Sandy; `2011-01-27`: 16 missing hours after a D.C. snowstorm; `2012-10-30`: 13 missing hours; `2011-01-18`: 12 missing hours). Missing hours are documented and never silently imputed with synthetic counts.
3. **Astronomical Season Coding Caveat**:
   - In both CSVs, `season` transitions on astronomical equinox/solstice dates (`Mar 21`, `Jun 21`, `Sep 23`, `Dec 21`), so `season = 1` (labeled `springer` in `Readme.txt`) covers ~Dec 21–Mar 20, `season = 2` (`summer`) covers ~Mar 21–Jun 20, `season = 3` (`fall`) covers ~Jun 21–Sep 22, and `season = 4` (`winter`) covers ~Sep 23–Dec 20. The dashboard and query engine document this and provide a sidebar toggle between official README labels and calendar labels.
4. **Sensor Edge Cases & Outlier Retention**:
   - All 22 hours on `2011-03-10` have `hum == 0.0` (sensor dropout, flagged in `reports/data_quality_report.md`).
   - 505 hourly counts exceed the global IQR fence (`cnt > 642.5`), and 438 exceed $|z_{\text{robust}}| > 3.5$ within `(hr, workingday)` cells. All are retained as genuine demand observations.
5. **Engineered Features & Leakage Prevention**:
   - De-normalized physical weather metrics (`temp_c`, `atemp_c`, `hum_pct`, `windspeed_kmh`).
   - Cyclical Fourier encodings (`hr_sin`, `hr_cos`, `weekday_sin`, `weekday_cos`, `mnth_sin`, `mnth_cos`).
   - Commuting indicators (`is_commute_hour` for `hr ∈ {7, 8, 16, 17, 18}` and interaction `workday_commute`).
   - `assert_no_leakage()` strictly blocks `casual`, `registered`, `cnt`, and `instant` from entering `MODEL_FEATURES`.

---

## 4. Hypothesis Formulation & Statistical Testing

Implemented in `src/hypothesis.py` (`reports/hypothesis_results.md`, `outputs/hypothesis_results.json`, `outputs/figures/hypothesis_commute.png`):

- **Research Question**: Is bike demand during commuting hours higher on working days than on non-working days?
- **Pre-Registered Commuting Windows**: Morning `07:00–08:59` (`hr ∈ {7, 8}`) and Evening `16:00–18:59` (`hr ∈ {16, 17, 18}`).
- **$H_0$**: $\mu_{\text{working}} = \mu_{\text{non-working}}$ for daily commuting-window rental totals.
- **$H_1$**: $\mu_{\text{working}} > \mu_{\text{non-working}}$ (one-sided, $\alpha = 0.05$).

### Key Statistical Results
- **Sample Sizes**: $n_{\text{working}} = 500$ days, $n_{\text{non-working}} = 231$ days.
- **Group Summaries**:
  - Working days: **Mean = 2,068.5**, **Median = 2,078.0**, $\text{SD} = 862.9$ rentals/day.
  - Non-working days: **Mean = 1,105.2**, **Median = 1,096.0**, $\text{SD} = 566.4$ rentals/day.
- **Difference in Means**: **+963.3 rentals/day** (ratio **1.87×**), Welch 95% CI **[858.0, 1,068.7]**, Bootstrap (10,000 resamples) 95% CI **[857.3, 1,069.2]**.
- **Primary Test (Welch's $t$-test)**: $t = 17.96$, $\text{df} = 645.6$, one-sided **$p = 4.33 \times 10^{-59}$** $\implies$ **Reject $H_0$**.
- **Effect Size**: **Cohen's $d = 1.23$, Hedges' $g = 1.23$** (very large).
- **Nonparametric Sensitivity (Mann-Whitney $U$)**: $U = 94,178$, one-sided **$p = 3.65 \times 10^{-43}$**, rank-biserial $r = 0.63$.
- **Confounding & Serial Correlation Adjustment (Newey-West HAC OLS, 7 lags)**: Controlling for quadratic temperature, humidity, wind speed, worst daily weather category, month fixed effects, and year (`R² = 0.682`), working days carry **1.89×** (95% CI **[1.77×, 2.02×]**, HAC $z = 19.50$, $p = 5.85 \times 10^{-85}$) the commuting-window demand of non-working days (linear adjusted difference **+927.9 rentals/day**, 95% CI **[848.0, 1,007.9]**).

---

## 5. Hourly Demand-Estimation Model

Implemented in `src/train_model.py` and `src/predict.py` (`reports/model_results.md`, `outputs/model_metrics.json`, `models/demand_model.joblib`):

### Validation & Split Design
- **Development Set**: `2011-01-01 00:00` to `2012-06-30 23:00` (**13,003 hours**).
- **Final Hold-Out Test Set**: `2012-07-01 00:00` to `2012-12-31 23:00` (**4,376 hours**), evaluated once after hyperparameter selection.
- **Hyperparameter Selection**: 4-fold expanding-window (rolling-origin) cross-validation inside the development period (`2011-Q3`, `2011-Q4`, `2012-Q1`, `2012-Q2`).
- **Nonnegative-Output Policy**: All model predictions are clipped at `0.0` (`np.clip(pred, 0, None)`), and the selected model uses a $\log(1 + y) / \exp(\hat{y}) - 1$ target transformation.

### Model Comparison & Hold-Out Test Performance

| Model | Dev CV RMSE | Dev CV MAE | Hold-Out Test MAE | Hold-Out Test RMSE | Hold-Out Test $R^2$ |
|---|---:|---:|---:|---:|---:|
| **Baseline (`hr × workingday` conditional mean)** | 105.22 | 73.46 | 107.21 | 150.35 | 0.532 |
| **Random Forest (`rf_log`)** | 77.96 | 52.80 | — | — | — |
| **HistGradientBoosting Poisson (`hgb_poisson`)** | 69.64 | 46.09 | — | — | — |
| **Selected: HistGradientBoosting Log (`hgb_log`)** | **68.62** | **45.78** | **43.54** | **68.18** | **0.904** |

- **Improvement over Baseline**: **-59.4% MAE reduction** and **-54.7% RMSE reduction** on the 4,376-hour held-out test set.
- **Empirical Interval Coverage**: Out-of-fold 10th–90th percentile residual bands calibrated on the development set cover **79.8%** of actual test hours.
- **Secondary Strict 2011 → 2012 Split**: Training on 2011 alone and testing on all of 2012 yields `MAE = 92.34`, `RMSE = 128.33`, $R^2 = 0.623$ (vs. baseline `MAE = 108.52`, $R^2 = 0.435$) due to the +64.9% year-over-year expansion that tree models cannot extrapolate without seeing early-2012 data.

---

## 6. Interactive Dashboard & Natural-Language Operations Prototype

Implemented in `app.py` and `src/query_engine.py`:

### Supported Natural-Language Queries (Deterministic Offline Engine + Optional LLM Fallback)
The natural-language assistant works completely **offline without any API key** via deterministic regex intent and entity extraction, and computes every numerical answer live from `hour.csv`, `outputs/hypothesis_results.json`, or `models/demand_model.joblib`:
- *"When is bike demand highest on working days?"* (`peak_hours`)
- *"Compare demand on working days and non-working days."* (`compare_day_types`)
- *"How does weather affect rental demand?"* (`weather_effect`)
- *"What is the average demand in summer?"* (`season_average`)
- *"Estimate bike rentals at 8 AM in summer on a working day."* (`predict`)
- *"Compare expected demand during clear weather and light rain."* (`predict`)
- *"Which hours should the operations team monitor most closely?"* (`monitor_hours`)
- *"Was the working-day difference statistically significant?"* (`hypothesis`)
- *"How accurate is the demand model?"* (`model_performance`)
- *"Which days had unusual demand?"* (`anomalies`)
- *"How did demand grow from 2011 to 2012?"* (`trend`)

Unrecognized questions never hallucinate answers; they politely decline and display supported examples. Optional environment variables (`BIKE_LLM_API_KEY`, `BIKE_LLM_BASE_URL`, `BIKE_LLM_MODEL`) can be set to map novel phrasings to one of the supported deterministic intent handlers without ever allowing the LLM to fabricate numbers.

---

## 7. Project Structure

```text
bike+sharing+dataset/
├── app.py                          # Interactive Streamlit operations dashboard & NL chat prototype
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation & reproduction guide
├── scripts/
│   └── download_data.py            # Verifies or downloads the official UCI Bike Sharing dataset
├── src/
│   ├── config.py                   # Central paths, constants & pre-registered commuting hours
│   ├── data_validation.py          # Programmatic data quality checks & cross-dataset verification
│   ├── features.py                 # Feature engineering & target-leakage assertions
│   ├── hypothesis.py               # Welch t-test, Mann-Whitney U, bootstrap CI & HAC OLS regression
│   ├── train_model.py              # Chronological CV, model selection, holdout evaluation & plots
│   ├── predict.py                  # Persisted model loader & scenario estimator with safety buffer
│   ├── analysis.py                 # Reusable descriptive summaries for dashboard & query engine
│   ├── query_engine.py             # Deterministic NL intent/entity parser & grounded answer builder
│   ├── anomaly.py                  # Conditional residual anomaly detector & D.C. event annotations
│   └── eda_plots.py                # Exploratory & data-completeness visualizations
├── tests/
│   └── test_project.py             # 23 automated pytest tests covering all pipeline stages
├── data/
│   ├── raw/                        # Original UCI hour.csv (17,379 rows) & day.csv (731 rows)
│   └── processed/                  # Engineered hour_features.csv
├── models/
│   ├── demand_model.joblib         # Deployed model bundle refit on 2011-2012
│   └── demand_model_eval.joblib    # Development-period model bundle (2011-01 to 2012-06)
├── outputs/
│   ├── data_quality.json           # Machine-readable validation results
│   ├── hypothesis_results.json     # Machine-readable hypothesis test metrics
│   ├── model_metrics.json          # Cross-validation & holdout evaluation metrics
│   ├── test_predictions.csv        # Holdout actual vs. predicted counts & 80% intervals
│   ├── anomalies.csv / .json       # Flagged conditional anomaly days
│   └── figures/                    # All generated PNG charts
├── reports/
│   ├── initial_audit.md            # Audit of existing R HTML/Rmd report & gap analysis
│   ├── data_quality_report.md      # Full data validation report
│   ├── hypothesis_results.md       # Statistical hypothesis testing report
│   ├── model_results.md            # Demand model evaluation & methodology report
│   └── final_findings.md           # Executive synthesis, findings, limitations & citations
└── time-series-master/             # Preserved original reference R HTML & Rmd files
```

---

## 8. Installation, Reproduction & Execution Commands

```powershell
# 1. Create and activate virtual environment (or use existing .venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Verify or download the official UCI dataset
python scripts/download_data.py

# 3. Run data quality checks and feature engineering
python -m src.data_validation
python -m src.features
python -m src.eda_plots

# 4. Run statistical hypothesis testing
python -m src.hypothesis

# 5. Train and evaluate the hourly demand-estimation model
python -m src.train_model

# 6. Run conditional anomaly detection
python -m src.anomaly

# 7. Execute the automated test suite (23 tests)
pytest -v

# 8. Launch the interactive Streamlit prototype
streamlit run app.py
```

---

## 9. Citations

- **Dataset**: Fanaee-T, H. (2013). *Bike Sharing Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
- **Methodology Paper**: Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence, 2(2–3), 113–127. https://doi.org/10.1007/s13748-013-0040-3.
- **Preserved Reference Report**: Hadi, R. (2018). *Bike Share: Time Series Analysis* (`time-series-analysis.Rmd` / `time-series-analysis.html`).

