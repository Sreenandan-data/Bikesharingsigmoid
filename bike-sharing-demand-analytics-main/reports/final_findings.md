# Final Analytical Findings, Experimental Results & Practical Implications

## 1. Executive Summary

This report synthesizes the end-to-end empirical findings from the **Capital Bikeshare (Washington, D.C., 2011–2012)** dataset (`hour.csv`: 17,379 hourly records; `day.csv`: 731 daily records). Across the two-year observation period, the system recorded **3,292,679 total bike rentals**, growing **+64.9%** year-over-year from **1,243,103 rentals in 2011** (mean **3,405.8/day**) to **2,049,576 rentals in 2012** (mean **5,599.9/day**). Registered subscribers account for **81.2%** of all rentals (`2,672,662`), while casual users account for **18.8%** (`620,017`).

All four core assignment deliverables—plus the conditional anomaly-detection extension—have been implemented, tested, and verified:
1. **Data Quality & Feature Engineering** (`src/data_validation.py`, `src/features.py`, `reports/data_quality_report.md`)
2. **Pre-Registered Hypothesis Testing** (`src/hypothesis.py`, `reports/hypothesis_results.md`)
3. **Time-Aware Hourly Demand Estimation** (`src/train_model.py`, `src/predict.py`, `reports/model_results.md`)
4. **Natural-Language Operations Prototype** (`app.py`, `src/query_engine.py`, `src/analysis.py`)
5. **Stretch Goal: Conditional Anomaly & Event Detection** (`src/anomaly.py`, `outputs/anomalies.csv`)

---

## 2. Data Quality & Feature Engineering Findings

### 2.1 Validation Summary
- **Completeness & Schema**: Both `hour.csv` (`17,379 × 17`) and `day.csv` (`731 × 16`) match the official UCI specification with **zero missing cell values (`NaN`)**, **zero duplicate rows**, **zero duplicate `instant` IDs**, and **strictly monotonic chronological ordering**.
- **Component Identity**: `cnt == casual + registered` holds exactly on **100% of rows** in both `hour.csv` (0 mismatches across 17,379 rows) and `day.csv` (0 mismatches across 731 rows).
- **Cross-Table Aggregation Consistency**: Summing `casual`, `registered`, and `cnt` in `hour.csv` by `dteday` matches `day.csv` with **zero discrepancy across all 731 dates**.
- **Missing Hourly Timestamps (165 Absent Hours)**: A complete 731-day hourly grid contains $731 \times 24 = 17,544$ hours, meaning **165 hourly timestamps are absent across 76 distinct calendar dates** (`655` days have all 24 hours; `62` have 23 hours; `14` have 22 or fewer hours). The largest gaps coincide with documented severe weather and system shutdowns:
  - `2012-10-29` (**23 missing hours**, 1 observed hour): Hurricane Sandy service suspension.
  - `2011-01-27` (**16 missing hours**, 8 observed hours): Post-blizzard recovery in Washington, D.C.
  - `2012-10-30` (**13 missing hours**, 11 observed hours): Hurricane Sandy afternoon reopening.
  - `2011-01-18` (**12 missing hours**, 12 observed hours): D.C. freezing rain / ice storm.
  - `2011-01-26` (**8 missing hours**, 16 observed hours) and `2011-08-28` (**7 missing hours**, Hurricane Irene).
- **Important Season Coding Finding**: Although `Readme.txt` labels `season` as `1:springer, 2:summer, 3:fall, 4:winter`, inspecting the transition dates (`2011-03-21`, `2011-06-21`, `2011-09-23`, `2011-12-21`) shows that the codes switch on **astronomical equinoxes/solstices** and are shifted by one season relative to common usage:
  - Code `1` ("spring" in README): ~Dec 21 – Mar 20 (astronomical winter; mean **2,604 rentals/day**)
  - Code `2` ("summer" in README): ~Mar 21 – Jun 20 (astronomical spring; mean **4,992 rentals/day**)
  - Code `3` ("fall" in README): ~Jun 21 – Sep 22 (astronomical summer; mean **5,644 rentals/day**)
  - Code `4` ("winter" in README): ~Sep 23 – Dec 20 (astronomical fall; mean **4,728 rentals/day**)
  Both the analytical reports and the Streamlit application expose a toggle between the official README labels and calendar-aligned labels so users are never misled.
- **Sensor Anomalies & Outliers**:
  - Exactly **22 hourly rows** (all on `2011-03-10`) record `hum == 0.0`, which is physically implausible and represents a weather-sensor dropout.
  - **505 hourly observations** exceed the global upper IQR fence (`cnt > 642.5`), and **438 observations** exceed $|z_{\text{robust}}| > 3.5$ conditional on `(hr, workingday)`. In contrast to the reference R report (`time-series-analysis.Rmd`), which used `forecast::tsclean()` to overwrite extreme counts before modeling, we **preserve all observations** because high commuting peaks and storm shutdowns are real operational regimes.

---

## 3. Hypothesis Testing Results

### 3.1 Primary Hypothesis: Commuting-Window Demand on Working vs. Non-Working Days
- **Research Question**: Is bike demand during commuting hours higher on working days than on non-working days?
- **Pre-Registered Commuting Windows**: Morning `hr ∈ {7, 8}` (`07:00–08:59`) and Evening `hr ∈ {16, 17, 18}` (`16:00–18:59`), totaling 5 hours per day.
- **Null Hypothesis ($H_0$)**: Mean daily commuting-window rental total is equal on working days (`workingday == 1`) and non-working days (`workingday == 0`).
- **Alternative Hypothesis ($H_1$)**: Mean daily commuting-window rental total is higher on working days ($\alpha = 0.05$, one-sided).

| Metric | Working Days (`workingday = 1`) | Non-Working Days (`workingday = 0`) | Difference / Test Result |
|---|---|---|---|
| **Sample Size ($n$ days)** | 500 | 231 | Total = 731 days |
| **Mean Commuting Rentals / Day** | **2,068.5** | **1,105.2** | **+963.3 rentals/day** (1.87× ratio) |
| **Median Commuting Rentals / Day** | **2,078.0** | **1,096.0** | **+982.0 rentals/day** |
| **Standard Deviation** | 862.9 | 566.4 | Levene $p = 6.34 \times 10^{-10}$ |
| **95% CI for Mean Difference** | — | — | **Welch: [858.0, 1,068.7]**; **Bootstrap: [857.3, 1,069.2]** |
| **Welch's $t$-test (one-sided)** | — | — | **$t = 17.96$, $\text{df} = 645.6$, $p = 4.33 \times 10^{-59}$** |
| **Mann-Whitney $U$ (one-sided)** | — | — | **$U = 94,178$, $p = 3.65 \times 10^{-43}$** (rank-biserial $r = 0.63$) |
| **Standardized Effect Size** | — | — | **Cohen's $d = 1.23$, Hedges' $g = 1.23$** (very large) |

### 3.2 Robustness to Serial Correlation, Weather Confounding & Incomplete Windows
Daily residuals exhibit strong lag-1 autocorrelation ($r_1 = 0.79$, Durbin-Watson $= 0.41$) due to seasonality and multi-year growth. To verify that the difference is not an artifact of serial correlation, missing hours, or weather differences:
- **Complete-Window Days Only** ($n = 495$ working vs. $230$ non-working days with all 5 commuting hours present): Mean difference **+977.9 rentals/day**, Welch $p = 3.89 \times 10^{-61}$, Hedges' $g = 1.27$.
- **Stratified by Year**:
  - **2011 only** ($n = 250$ vs. $115$): Mean difference **+677.0 rentals/day**, Welch $p = 1.48 \times 10^{-29}$, Hedges' $g = 1.24$.
  - **2012 only** ($n = 250$ vs. $116$): Mean difference **+1,251.8 rentals/day**, Welch $p = 2.87 \times 10^{-46}$, Hedges' $g = 1.77$.
- **OLS with Newey-West HAC Standard Errors (7 Lags) Controlling for Weather, Month & Year**:
  Estimating $\log(1 + \text{commute\_total}) \sim \text{workingday} + \text{temp} + \text{temp}^2 + \text{hum} + \text{windspeed} + C(\text{weathersit\_max}) + C(\text{mnth}) + \text{yr}$ ($R^2 = 0.682$) yields a working-day multiplier of **1.89×** (95% CI **[1.77×, 2.02×]**, HAC $z = 19.50$, $p = 5.85 \times 10^{-85}$) and a linear-scale adjusted difference of **+927.9 rentals/day** (95% CI **[848.0, 1,007.9]**).

### 3.3 Secondary Hypothesis: Precipitation Impact on Daily Demand
- Days with at least one hour of light rain/snow or worse (`weathersit >= 3`, $n = 300$) average **4,014 rentals/day** (median **4,101**) versus **4,846 rentals/day** (median **4,795**) on dry/mist-only days ($n = 431$): Welch one-sided $t = 5.86$, $p = 3.02 \times 10^{-9}$; Mann-Whitney $p = 1.24 \times 10^{-8}$; mean within-season-year difference **-829 rentals/day**.

---

## 4. Hourly Demand-Estimation Model Results

### 4.1 Experimental Design & Leakage Prevention
- **Target**: Hourly `cnt`.
- **Excluded Columns**: `casual` and `registered` (which sum identically to `cnt`) and `instant` (record index) are strictly excluded, enforced by automated assertion (`src/features.py:assert_no_leakage`).
- **Primary Chronological Split**:
  - **Development Set**: `2011-01-01 00:00` to `2012-06-30 23:00` (**13,003 hours**, 18 months).
  - **Held-Out Test Set**: `2012-07-01 00:00` to `2012-12-31 23:00` (**4,376 hours**, final 6 months).
- **Model Selection**: 4-fold expanding-window (rolling-origin) cross-validation strictly inside the development set (`2011-Q3`, `2011-Q4`, `2012-Q1`, `2012-Q2`).
- **Why 18-Month Development Beats a Pure 2011 → 2012 Split**: We also ran a strict secondary experiment training exclusively on 2011 (`yr == 0`) and testing on all of 2012 (`yr == 1`). Because system ridership jumped +64.9% between 2011 and 2012 and tree ensembles cannot observe the `yr = 1` regime shift when trained only on `yr = 0`, the 2011-only model suffers a systematic negative bias of **-89.65 rentals/hour** on 2012 (`MAE = 92.34`, `RMSE = 128.33`, $R^2 = 0.623$). Including the first half of 2012 in the development set allows the model to learn the 2012 demand level while still evaluating strictly out-of-time on the second half of 2012.

### 4.2 Quantitative Performance Summary

| Model | Dev CV Mean RMSE | Dev CV Mean MAE | Test MAE (Jul–Dec 2012) | Test RMSE (Jul–Dec 2012) | Test $R^2$ | Test Mean Bias |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (`hr × workingday` mean)** | 105.22 | 73.46 | 107.21 | 150.35 | 0.532 | -87.26 |
| **Random Forest (`rf_log`)** | 77.96 | 52.80 | — | — | — | — |
| **HistGradientBoosting (`hgb_poisson`)** | 69.64 | 46.09 | — | — | — | — |
| **Selected: `hgb_log` ($\log(1+y)$ HGB)** | **68.62** | **45.78** | **43.54** | **68.18** | **0.904** | **-26.11** |

- **Error Reduction vs. Baseline**: On the held-out test set, the selected `hgb_log` model reduces hourly **MAE by 59.4%** (`107.21 → 43.54`) and **RMSE by 54.7%** (`150.35 → 68.18`), raising $R^2$ from **0.532 to 0.904**.
- **Aggregated Daily Totals on Test Set**: Summing hourly predictions across each day in Jul–Dec 2012 yields a daily **MAE of 806 rentals/day** ($R^2 = 0.724$) versus **2,408 rentals/day** ($R^2 = -1.327$) for the baseline.
- **Indicative 80% Prediction Interval Calibration**: Out-of-fold residual quantiles (10th–90th percentiles binned by predicted demand level on the development set) achieve **79.8% empirical coverage** on the held-out test set (target: 80.0%).
- **Top Drivers (Test Set Permutation Importance, RMSE increase)**: `hr` (+131.8), `workday_commute` (+59.4), `hr_sin` (+38.6), `hr_cos` (+33.0), `temp` (+23.0), `hum` (+19.3), and `yr` (+18.9).

---

## 5. Conditional Anomaly & Event Detection Findings

Using 8-fold contiguous block cross-fitting on the 731 daily records (`src/anomaly.py`), we compared actual daily rentals against conditional expected demand given season, year, month, holiday, weekday, working-day status, and weather (`|z_robust| >= 3.0` or shortfall `<= -60%` when expected `>= 1,500`). Exactly **16 candidate anomaly days** (11 low, 5 high) were flagged:
- **Hurricane Sandy (`2012-10-29` and `2012-10-30`)**: On October 29, 2012, observed rentals dropped to **22** versus **2,451 expected** (**-99.1%**, `robust_z = -2.39`, only 1 hour logged before full system suspension). On October 30, observed rentals were **1,096** versus **2,855 expected** (**-61.6%**, 11 hours logged as service resumed in the afternoon).
- **Winter Storms & Ice Events**: `2011-01-27` (**431 actual vs. 2,418 expected**, -82.2%, 8 hours observed after the Jan 26 D.C. snowstorm) and `2011-01-18` (**683 actual vs. 1,750 expected**, -61.0%, 12 hours observed during a freezing-rain storm).
- **Unmodeled Calendar / Holiday Effects**: `2012-12-24` (Christmas Eve, coded as `workingday = 1`, **920 actual vs. 3,115 expected**, -70.5%) and `2012-12-26` (**441 actual vs. 1,648 expected**, -73.2%).
- **Cherry Blossom & St. Patrick's Day Surge (`2012-03-17` to `2012-03-20`)**: On Saturday, March 17, 2012 (`temp = 21.1 °C`), rentals reached **7,836** versus **2,187 expected** (**+258.2%**, `robust_z = +5.18`), coinciding with St. Patrick's Day and the opening weekend of the National Cherry Blossom Festival.

---

## 6. Operational Implications & Limitations

1. **Dual-Regime Fleet Scheduling**:
   - On **working days**, operations must prioritize pre-positioning bikes ahead of the **08:00 AM peak** (mean **477 rentals/hr**, 90th percentile **692/hr**) and turn-around rebalancing ahead of the **17:00–18:00 PM peak** (mean **525/hr** and **492/hr**, 90th percentile **782/hr** and **731/hr**).
   - On **non-working days**, pressure shifts to a broad **12:00–16:00 midday recreational plateau** (mean **366–372/hr**, 90th percentile **590–605/hr**) with higher sensitivity to pleasant weather (`weathersit = 1`).
2. **Distinction Between System Rental Demand and Station Inventory**:
   - Our model predicts **system-wide hourly checkout volume (`cnt`)**. Translating checkout volume into physical bikes required at individual stations requires origin–destination trip matrices, trip duration distributions, dock capacity constraints, and rebalancing truck routes—none of which are present in the aggregated UCI dataset.
   - In the Streamlit prototype, the **safety planning buffer (default +15%)** is explicitly separated from the statistical point estimate and labeled as a user-configurable operational assumption.
3. **Key Limitations & Future Improvements**:
   - **No autoregressive lag features in the what-if model**: Because operations managers use the model to query arbitrary future or hypothetical scenarios without supplying the prior hour's actual count, hourly residuals remain serially correlated ($\text{lag-1 } r = 0.58$). For pure 1-step-ahead automated dispatch, adding `cnt_lag1`, `cnt_lag24`, and `cnt_rolling_24h` would further reduce short-horizon RMSE.
   - **Regime shifts beyond 2012**: Tree-based models cannot extrapolate secular fleet expansion beyond the training range; continuous production deployment requires periodic rolling retraining or a trend-extrapolating state-space/linear component.
   - **Sparse extreme weather (`weathersit = 4`)**: Only 3 hours in the entire two-year dataset are coded as `weathersit = 4` (heavy rain/snow/ice pellets), making statistical estimates for that category inherently high-variance.

---

## 7. Citations & Attribution

1. **Official Dataset**: Fanaee-T, H. (2013). *Bike Sharing Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
2. **Primary Dataset Publication**: Fanaee-T, H., & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in Artificial Intelligence, 2(2–3), 113–127. https://doi.org/10.1007/s13748-013-0040-3.
3. **Reference Workspace Report**: Hadi, R. (2018). *Bike Share: Time Series Analysis* (`time-series-analysis.Rmd` / `time-series-analysis.html`), preserved in `time-series-master/` and audited in `reports/initial_audit.md`.
4. **Historical Weather & Operational Events**: Capital Bikeshare / ARLnow service advisories for Hurricane Irene (August 27–28, 2011) and Hurricane Sandy (October 29–30, 2012) in Washington, D.C.

