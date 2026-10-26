# Initial Workspace Audit

## 1. Existing Workspace Structure & Files

Prior to modification, the workspace contained:
- `day.csv` (57,569 bytes; SHA-256 `A6BCF826782D3C0FBFDCBEEAD17CD0884185A0DAFE8FF10CD48A874EE7BA18BE`): 731 daily records of the original UCI Capital Bikeshare dataset (2011–2012).
- `hour.csv` (1,156,736 bytes; SHA-256 `E03DE4EE4EF4DC376AC6E04BF829673C6269E8EBA5C60FA121640FA2F829504F`): 17,379 hourly records of the original UCI Capital Bikeshare dataset (2011–2012).
- `Readme.txt`: The official dataset documentation by Hadi Fanaee-T and João Gama (2013).
- `time-series-master/time-series-master/`: Reference directory containing:
  - `time-series-analysis.html` (3.26 MB): Rendered HTML output of an R Markdown blog post.
  - `time-series-analysis.Rmd` (9.5 KB): Source R Markdown file authored by **Rebecca Hadi** (dated `12/16/2018`), titled *"Bike Share: Time Series Analysis"*, reproducing an introductory tutorial on ARIMA forecasting in R (`forecast`, `tseries`, `ggplot2`).
  - Duplicate copies of `day.csv`, `hour.csv`, and `Readme.txt`.

All original files (`time-series-analysis.html`, `time-series-analysis.Rmd`, `day.csv`, `hour.csv`, `Readme.txt`) have been preserved untouched, and working copies of the raw data have been staged under `data/raw/`.

---

## 2. Review of the Existing Reference Analysis (`time-series-analysis.Rmd` / `.html`)

The existing R analysis performs a univariate daily time-series walkthrough on `day.csv`:
1. **Outlier cleaning (`forecast::tsclean`)**: Automatically strips and interpolates extreme daily counts prior to analysis.
2. **Smoothing (`forecast::ma`)**: Computes 7-day and 30-day moving averages on the cleaned daily series.
3. **Seasonal decomposition (`stl`)**: Applies additive STL decomposition (`frequency = 30`) to the 7-day moving average and extracts a seasonally adjusted series (`seasadj`).
4. **Stationarity testing (`tseries::adf.test`)**: Runs Augmented Dickey-Fuller tests before and after first-order differencing (`d = 1`), plus ACF/PACF inspection.
5. **ARIMA fitting & holdout plot**: Fits `auto.arima` (`ARIMA(1,1,1)`), manually specifies `ARIMA(1,1,7)` based on lag-7 autocorrelation, and plots a 25-day holdout (`start = 700` to `725`) on the pre-smoothed, deseasonalized series.

### Key Limitations & Risks in the Reference R Report
- **Daily univariate scope only**: It ignores `hour.csv` (17,379 rows) and never models hourly rental demand, commuting peaks, or exogenous weather/calendar predictors (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`, `workingday`, `hr`).
- **Pre-split data leakage in time-series preprocessing**: `tsclean()`, `ma(order=7)`, and `stl()` are applied across the *entire* 731-day series before slicing the holdout window (`700:725`). Because centered moving averages and STL smooth across future observations, information from the holdout window leaks into the pre-processed values.
- **Silent removal of genuine operational events**: Using `tsclean()` automatically erases extreme demand drops (such as the October 29–30, 2012 Hurricane Sandy shutdown) as "outliers" rather than recognizing them as real weather/event phenomena.
- **No quantitative holdout accuracy metrics**: Models are compared only via in-sample AIC/BIC on the transformed series; no out-of-sample MAE, RMSE, or $R^2$ is computed in actual rental count units.
- **Season label pitfall not documented**: In the UCI dataset, `season` switches on astronomical equinox/solstice dates (`1` = ~Dec 21–Mar 20, `2` = ~Mar 21–Jun 20, `3` = ~Jun 21–Sep 22, `4` = ~Sep 23–Dec 20), even though `Readme.txt` labels `1:springer, 2:summer, 3:fall, 4:winter`.

---

## 3. Gap Analysis Against Official Assignment Requirements

| Deliverable | Status in Existing Workspace | Required Implementation |
|---|---|---|
| **1. Data Quality & Feature Engineering** | Missing (only ad-hoc `tsclean` on `day.csv`) | Full programmatic validation of `hour.csv` (17,379 rows) and `day.csv` (731 rows), cross-dataset consistency checks, gap analysis (165 missing hourly timestamps across 76 dates), outlier flagging without silent deletion, cyclical encodings, and leakage-free feature pipeline (`src/data_validation.py`, `src/features.py`). |
| **2. Hypothesis Formulation & Testing** | Missing (only ADF stationarity test) | Pre-registered commuting-window hypothesis test comparing working vs. non-working days (`07:00–08:59` and `16:00–18:59`), Welch's t-test, Mann-Whitney U sensitivity check, bootstrap CI, and Newey-West HAC regression controlling for weather, seasonality, and serial correlation (`src/hypothesis.py`). |
| **3. Hourly Demand-Estimation Model** | Missing (daily univariate ARIMA only) | Time-aware chronological validation and holdout testing on `hour.csv` comparing a conditional historical baseline (`hr × workingday`) against tree-based regression (`HistGradientBoostingRegressor` and `RandomForestRegressor`), nonnegative prediction clipping, residual diagnostics, and operational interpretation (`src/train_model.py`, `src/predict.py`). |
| **4. Natural-Language Operations Prototype** | Missing | Interactive Streamlit application (`app.py`) backed by a deterministic intent and entity extraction engine (`src/query_engine.py`) with optional environment-configured LLM intent fallback, interactive visualizations, hypothesis viewer, and scenario estimator. |
| **Stretch: Anomaly Detection** | Missing (`tsclean` discarded extremes without analysis) | Cross-fitted conditional expectation residual detector identifying unusual daily demand while controlling for weather, calendar, and system growth (`src/anomaly.py`). |

---

## 4. Attribution & Reuse Policy

- The reference files `time-series-analysis.Rmd` and `time-series-analysis.html` are credited to **Rebecca Hadi (2018)** and the underlying tutorial she cited. None of her text or R code is presented as original work here.
- Concepts from classical time-series analysis (inspecting trend/seasonality, checking residual autocorrelation, and evaluating on a chronological holdout rather than a random split) are carried forward into a clean, leak-free Python implementation built from scratch around `hour.csv` and `day.csv`.

