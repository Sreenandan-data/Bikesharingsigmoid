"""Train and evaluate the hourly demand-estimation model.

Experimental design (time-aware, no random splitting):

* Development period: 2011-01-01 .. 2012-06-30.  Final hold-out test: 2012-07-01 .. 2012-12-31.
* Model selection uses rolling-origin (expanding-window) validation inside the development period
  only: four 3-month validation blocks (2011 Q3, 2011 Q4, 2012 Q1, 2012 Q2), each scored by a model
  trained on all earlier data.  The test period is touched once, after the configuration is fixed.
* A secondary "strict" experiment trains on 2011 only and tests on all of 2012, to quantify how
  badly a model extrapolates the system's year-over-year growth when it has never seen 2012.
* The deployed model (used by the app) is the selected configuration refit on all 2011-2012 data.

Run: ``python -m src.train_model``
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.features import MODEL_FEATURES, TARGET, assert_no_leakage, build_features

DEV_END = pd.Timestamp("2012-06-30 23:00")
TEST_START = pd.Timestamp("2012-07-01 00:00")
CV_FOLDS = [  # (validation start, validation end) - training uses everything before start
    ("2011-07-01", "2011-09-30"),
    ("2011-10-01", "2011-12-31"),
    ("2012-01-01", "2012-03-31"),
    ("2012-04-01", "2012-06-30"),
]


# ----------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, test_start: pd.Timestamp = TEST_START):
    """Return (development, test) frames split strictly by time."""
    dev = df[df["datetime"] < test_start].copy()
    test = df[df["datetime"] >= test_start].copy()
    return dev, test


def rolling_origin_folds(df: pd.DataFrame, folds=CV_FOLDS):
    """Yield (name, train_idx, val_idx) with training data strictly before validation data."""
    for start, end in folds:
        s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
        train_idx = df.index[df["datetime"] < s]
        val_idx = df.index[(df["datetime"] >= s) & (df["datetime"] <= e)]
        yield f"{start}..{end}", train_idx, val_idx


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
class HourWorkingdayBaseline:
    """Historical mean of cnt conditional on (hour, workingday), learned from training data."""

    __module__ = "src.train_model"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        t = pd.DataFrame({"hr": X["hr"].to_numpy(), "workingday": X["workingday"].to_numpy(),
                          "y": np.asarray(y)})
        self.table_ = t.groupby(["hr", "workingday"])["y"].mean()
        self.global_ = float(t["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([X["hr"].to_numpy(), X["workingday"].to_numpy()])
        return self.table_.reindex(keys).fillna(self.global_).to_numpy()


def make_model(family: str, params: dict):
    if family == "hgb_log":
        est = HistGradientBoostingRegressor(loss="squared_error", max_iter=600, early_stopping=False,
                                            random_state=config.RANDOM_STATE, **params)
        return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
    if family == "hgb_poisson":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=600, early_stopping=False,
                                             random_state=config.RANDOM_STATE, **params)
    if family == "rf_log":
        est = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=config.RANDOM_STATE,
                                    **params)
        return TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
    if family == "baseline":
        return HourWorkingdayBaseline()
    raise ValueError(family)


def search_space() -> list[tuple[str, dict]]:
    space: list[tuple[str, dict]] = [("baseline", {})]
    for fam in ("hgb_log", "hgb_poisson"):
        for lr, leaves, msl in itertools.product([0.05, 0.1], [31, 63], [20, 50]):
            space.append((fam, {"learning_rate": lr, "max_leaf_nodes": leaves,
                                "min_samples_leaf": msl, "l2_regularization": 1.0}))
    for mf, msl in itertools.product([0.5, 1.0], [1, 5]):
        space.append(("rf_log", {"max_features": mf, "min_samples_leaf": msl}))
    return space


def predict_counts(model, X: pd.DataFrame) -> np.ndarray:
    """Nonnegative-output policy: predictions are clipped at 0 (a count cannot be negative)."""
    return np.clip(np.asarray(model.predict(X[MODEL_FEATURES]), dtype=float), 0, None)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
            "mean_actual": float(y_true.mean()), "mean_predicted": float(y_pred.mean()),
            "bias": float((y_pred - y_true).mean()), "n": int(len(y_true))}


def fit(family: str, params: dict, X: pd.DataFrame, y: pd.Series):
    Xf = X[MODEL_FEATURES]
    assert_no_leakage(list(Xf.columns))
    return make_model(family, params).fit(Xf, y)


# ----------------------------------------------------------------------------------------------
# Cross-validation and selection
# ----------------------------------------------------------------------------------------------
@dataclass
class CVResult:
    family: str
    params: dict
    fold_rmse: list
    fold_mae: list

    @property
    def mean_rmse(self) -> float:
        return float(np.mean(self.fold_rmse))

    @property
    def mean_mae(self) -> float:
        return float(np.mean(self.fold_mae))


def cross_validate(dev: pd.DataFrame, verbose: bool = True) -> tuple[list[CVResult], dict]:
    results: list[CVResult] = []
    folds = list(rolling_origin_folds(dev))
    oof: dict[str, pd.Series] = {}
    for family, params in search_space():
        rmses, maes = [], []
        preds = pd.Series(np.nan, index=dev.index)
        for _, tr, va in folds:
            m = fit(family, params, dev.loc[tr], dev.loc[tr, TARGET])
            p = predict_counts(m, dev.loc[va])
            preds.loc[va] = p
            rmses.append(float(np.sqrt(mean_squared_error(dev.loc[va, TARGET], p))))
            maes.append(float(mean_absolute_error(dev.loc[va, TARGET], p)))
        r = CVResult(family, params, rmses, maes)
        results.append(r)
        oof[json.dumps([family, params], sort_keys=True)] = preds
        if verbose:
            print(f"  {family:12s} {params} -> CV RMSE {r.mean_rmse:7.2f}  MAE {r.mean_mae:7.2f}")
    return results, oof


def residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 5) -> dict:
    """Empirical 10th/90th percentiles of actual/predicted ratio per predicted-demand bin.

    Used to show an indicative 80% range around a prediction. Estimated from out-of-fold
    (development-period) predictions only."""
    d = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    edges = np.unique(np.quantile(d["p"], np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    d["bin"] = pd.cut(d["p"], edges, labels=False)
    out = []
    for b, g in d.groupby("bin"):
        res = g["y"] - g["p"]
        out.append({"lower_edge": float(edges[int(b)]), "upper_edge": float(edges[int(b) + 1]),
                    "q10_residual": float(res.quantile(0.10)), "q90_residual": float(res.quantile(0.90)),
                    "n": int(len(g))})
    return {"bins": out}


def interval_for(pred: np.ndarray, rq: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, float)
    lo, hi = np.empty_like(pred), np.empty_like(pred)
    for b in rq["bins"]:
        m = (pred > b["lower_edge"]) & (pred <= b["upper_edge"])
        lo[m] = pred[m] + b["q10_residual"]
        hi[m] = pred[m] + b["q90_residual"]
    return np.clip(lo, 0, None), np.clip(hi, 0, None)


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def make_plots(test: pd.DataFrame, pred: np.ndarray, base: np.ndarray, importances: pd.Series) -> list[str]:
    fd = config.FIGURES_DIR
    paths = []
    y = test[TARGET].to_numpy()
    resid = y - pred

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(pred, y, s=4, alpha=0.3)
    lim = max(y.max(), pred.max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Predicted rentals / hour")
    ax.set_ylabel("Actual rentals / hour")
    ax.set_title("Actual vs predicted (test: Jul-Dec 2012)")
    fig.tight_layout(); fig.savefig(fd / "model_actual_vs_predicted.png", dpi=130); plt.close(fig)
    paths.append("model_actual_vs_predicted.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].scatter(pred, resid, s=4, alpha=0.3)
    axes[0].axhline(0, color="r", lw=1)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predicted")
    axes[1].hist(resid, bins=80)
    axes[1].set_title("Residual distribution"); axes[1].set_xlabel("Residual")
    t = test.assign(abs_err_model=np.abs(resid), abs_err_base=np.abs(y - base))
    byh = t.groupby("hr")[["abs_err_model", "abs_err_base"]].mean()
    axes[2].plot(byh.index, byh["abs_err_base"], marker="o", label="Baseline (hour x workingday mean)")
    axes[2].plot(byh.index, byh["abs_err_model"], marker="o", label="Selected model")
    axes[2].set_xlabel("Hour of day"); axes[2].set_ylabel("MAE")
    axes[2].set_title("Prediction error by hour (test)"); axes[2].legend()
    fig.tight_layout(); fig.savefig(fd / "model_residuals.png", dpi=130); plt.close(fig)
    paths.append("model_residuals.png")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    w = test[(test["datetime"] >= "2012-09-03") & (test["datetime"] < "2012-09-17")]
    idx = w.index
    pos = test.index.get_indexer(idx)
    axes[0].plot(w["datetime"], w[TARGET], label="Actual", lw=1.2)
    axes[0].plot(w["datetime"], pred[pos], label="Model", lw=1.2)
    axes[0].plot(w["datetime"], base[pos], label="Baseline", lw=1, alpha=0.7)
    axes[0].set_title("Hourly demand, two test weeks (3-16 Sep 2012)"); axes[0].legend()
    daily = t.assign(pred=pred, base=base).groupby("date")[[TARGET, "pred", "base"]].sum()
    axes[1].plot(daily.index, daily[TARGET], label="Actual", lw=1.2)
    axes[1].plot(daily.index, daily["pred"], label="Model", lw=1.2)
    axes[1].plot(daily.index, daily["base"], label="Baseline", lw=1, alpha=0.7)
    axes[1].set_title("Daily totals of hourly predictions (test period)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(fd / "model_time_series.png", dpi=130); plt.close(fig)
    paths.append("model_time_series.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    imp = importances.sort_values()
    ax.barh(imp.index, imp.values)
    ax.set_title("Permutation importance (test period, increase in RMSE)")
    fig.tight_layout(); fig.savefig(fd / "model_permutation_importance.png", dpi=130); plt.close(fig)
    paths.append("model_permutation_importance.png")
    return paths


# ----------------------------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------------------------
def run(df: pd.DataFrame | None = None, save: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    if df is None:
        from src.data_validation import load_raw
        df = build_features(load_raw(config.HOUR_CSV))
    df = df.sort_values("datetime").reset_index(drop=True)
    assert_no_leakage(MODEL_FEATURES)
    dev, test = chronological_split(df)
    assert dev["datetime"].max() < test["datetime"].min()

    if verbose:
        print(f"Development rows {len(dev):,} ({dev.datetime.min()} .. {dev.datetime.max()}); "
              f"test rows {len(test):,} ({test.datetime.min()} .. {test.datetime.max()})")
    cv, oof = cross_validate(dev, verbose=verbose)
    ml = [r for r in cv if r.family != "baseline"]
    best = min(ml, key=lambda r: r.mean_rmse)
    base_cv = next(r for r in cv if r.family == "baseline")
    best_by_family = {}
    for r in cv:
        if r.family not in best_by_family or r.mean_rmse < best_by_family[r.family].mean_rmse:
            best_by_family[r.family] = r

    # Out-of-fold residual quantiles for indicative intervals (development data only).
    oof_pred = oof[json.dumps([best.family, best.params], sort_keys=True)]
    mask = oof_pred.notna()
    rq = residual_quantiles(dev.loc[mask, TARGET].to_numpy(), oof_pred[mask].to_numpy())

    # Final evaluation: refit on development period, score once on test period.
    model = fit(best.family, best.params, dev, dev[TARGET])
    baseline = fit("baseline", {}, dev, dev[TARGET])
    pred = predict_counts(model, test)
    base = predict_counts(baseline, test)
    y = test[TARGET].to_numpy()
    lo, hi = interval_for(pred, rq)
    coverage = float(((y >= lo) & (y <= hi)).mean())
    test_metrics = {"model": metrics(y, pred), "baseline": metrics(y, base)}
    td = test.assign(pred=pred, base=base).groupby("date")[[TARGET, "pred", "base"]].sum()
    daily_metrics = {"model": metrics(td[TARGET], td["pred"]),
                     "baseline": metrics(td[TARGET], td["base"])}

    # Strict secondary experiment: train on 2011, test on 2012.
    tr11, te12 = df[df["yr"] == 0], df[df["yr"] == 1]
    m11 = fit(best.family, best.params, tr11, tr11[TARGET])
    b11 = fit("baseline", {}, tr11, tr11[TARGET])
    strict = {"model": metrics(te12[TARGET], predict_counts(m11, te12)),
              "baseline": metrics(te12[TARGET], predict_counts(b11, te12))}

    perm = permutation_importance(model, test[MODEL_FEATURES], y, n_repeats=5,
                                  random_state=config.RANDOM_STATE,
                                  scoring="neg_root_mean_squared_error", n_jobs=1)
    importances = pd.Series(perm.importances_mean, index=MODEL_FEATURES)

    # Deployed model: same configuration, refit on all data.
    final_model = fit(best.family, best.params, df, df[TARGET])
    final_baseline = fit("baseline", {}, df, df[TARGET])
    a = np.polyfit(df["temp"] * 41, df["atemp"] * 50, 1)

    results = {
        "design": {
            "development_period": [str(dev.datetime.min()), str(dev.datetime.max())],
            "test_period": [str(test.datetime.min()), str(test.datetime.max())],
            "n_development": int(len(dev)), "n_test": int(len(test)),
            "cv_folds": CV_FOLDS, "selection_metric": "mean RMSE over rolling-origin folds",
            "features": MODEL_FEATURES, "target": TARGET,
            "excluded_for_leakage": ["casual", "registered", "instant"],
            "nonnegative_policy": "predictions clipped at 0",
        },
        "cv_results": [{"family": r.family, "params": r.params, "fold_rmse": r.fold_rmse,
                        "fold_mae": r.fold_mae, "mean_rmse": r.mean_rmse, "mean_mae": r.mean_mae}
                       for r in cv],
        "cv_best_by_family": {k: {"params": v.params, "mean_rmse": v.mean_rmse, "mean_mae": v.mean_mae}
                              for k, v in best_by_family.items()},
        "selected": {"family": best.family, "params": best.params, "cv_mean_rmse": best.mean_rmse,
                     "cv_mean_mae": best.mean_mae, "baseline_cv_mean_rmse": base_cv.mean_rmse},
        "test_metrics_hourly": test_metrics,
        "test_metrics_daily_totals": daily_metrics,
        "test_improvement_vs_baseline": {
            "MAE_reduction_pct": 100 * (1 - test_metrics["model"]["MAE"] / test_metrics["baseline"]["MAE"]),
            "RMSE_reduction_pct": 100 * (1 - test_metrics["model"]["RMSE"] / test_metrics["baseline"]["RMSE"]),
        },
        "interval_80pct_empirical_test_coverage": coverage,
        "residual_quantiles": rq,
        "strict_2011_train_2012_test": strict,
        "permutation_importance_test": importances.sort_values(ascending=False).to_dict(),
        "test_residual_summary": {
            "mean": float((y - pred).mean()), "median": float(np.median(y - pred)),
            "p05": float(np.percentile(y - pred, 5)), "p95": float(np.percentile(y - pred, 95)),
            "lag1_autocorrelation": float(pd.Series(y - pred).autocorr(1)),
        },
        "atemp_from_temp": {"slope": float(a[0]), "intercept": float(a[1])},
        "runtime_seconds": round(time.time() - t0, 1),
    }

    if save:
        config.ensure_dirs()
        results["figures"] = make_plots(test, pred, base, importances)
        bundle = {
            "model": final_model, "baseline": final_baseline, "features": MODEL_FEATURES,
            "family": best.family, "params": best.params, "trained_on": "2011-01-01..2012-12-31 (all data)",
            "residual_quantiles": rq, "atemp_from_temp": results["atemp_from_temp"],
            "test_metrics_hourly": test_metrics, "nonnegative_policy": "clip at 0",
        }
        joblib.dump(bundle, config.MODELS_DIR / "demand_model.joblib", compress=3)
        joblib.dump({"model": model, "baseline": baseline, "features": MODEL_FEATURES,
                     "trained_on": "2011-01-01..2012-06-30 (development period)"},
                    config.MODELS_DIR / "demand_model_eval.joblib", compress=3)
        (config.OUTPUTS_DIR / "model_metrics.json").write_text(json.dumps(results, indent=2))
        test.assign(predicted=pred, baseline=base, lower80=lo, upper80=hi)[
            ["datetime", "hr", "workingday", "weathersit", TARGET, "predicted", "baseline", "lower80", "upper80"]
        ].to_csv(config.OUTPUTS_DIR / "test_predictions.csv", index=False)
        (config.REPORTS_DIR / "model_results.md").write_text(to_markdown(results), encoding="utf-8")
    if verbose:
        tm = test_metrics
        print(f"Selected {best.family} {best.params}")
        print(f"TEST model    MAE {tm['model']['MAE']:.2f} RMSE {tm['model']['RMSE']:.2f} R2 {tm['model']['R2']:.3f}")
        print(f"TEST baseline MAE {tm['baseline']['MAE']:.2f} RMSE {tm['baseline']['RMSE']:.2f} R2 {tm['baseline']['R2']:.3f}")
        print(f"Runtime {results['runtime_seconds']} s")
    return results


def to_markdown(r: dict) -> str:
    tm, dm, s = r["test_metrics_hourly"], r["test_metrics_daily_totals"], r["strict_2011_train_2012_test"]
    sel = r["selected"]
    lines = [
        "# Demand Model Results",
        "",
        "Generated by `python -m src.train_model`. Every number is produced by that script run; nothing is hand-entered.",
        "",
        "## Design",
        "",
        f"- Target: hourly `cnt`. Features ({len(r['design']['features'])}): `{', '.join(r['design']['features'])}`.",
        "- Excluded for leakage: `casual`, `registered` (sum to the target) and `instant` (row index).",
        f"- Development period: {r['design']['development_period'][0]} to {r['design']['development_period'][1]} "
        f"({r['design']['n_development']:,} hours).",
        f"- Final test period: {r['design']['test_period'][0]} to {r['design']['test_period'][1]} "
        f"({r['design']['n_test']:,} hours), used once after selection.",
        "- Model selection: rolling-origin validation with four 3-month blocks inside the development period "
        "(each block predicted by a model trained only on earlier data); lowest mean RMSE wins.",
        "- Nonnegative-output policy: predictions are clipped at 0.",
        "",
        "## Cross-validation (development period only)",
        "",
        "| Family | Best params | CV mean RMSE | CV mean MAE |",
        "|---|---|---|---|",
    ]
    for fam, v in r["cv_best_by_family"].items():
        lines.append(f"| {fam} | `{v['params']}` | {v['mean_rmse']:.2f} | {v['mean_mae']:.2f} |")
    lines += [
        "",
        f"**Selected:** `{sel['family']}` with `{sel['params']}` (CV RMSE {sel['cv_mean_rmse']:.2f} vs "
        f"baseline {sel['baseline_cv_mean_rmse']:.2f}). Full grid: `outputs/model_metrics.json`.",
        "",
        "## Final hold-out test (Jul-Dec 2012)",
        "",
        "| Model | MAE | RMSE | R^2 | Mean bias |",
        "|---|---|---|---|---|",
        f"| Baseline: mean by hour x working day | {tm['baseline']['MAE']:.2f} | {tm['baseline']['RMSE']:.2f} | "
        f"{tm['baseline']['R2']:.3f} | {tm['baseline']['bias']:.2f} |",
        f"| Selected model ({sel['family']}) | {tm['model']['MAE']:.2f} | {tm['model']['RMSE']:.2f} | "
        f"{tm['model']['R2']:.3f} | {tm['model']['bias']:.2f} |",
        "",
        f"- Improvement over baseline: MAE -{r['test_improvement_vs_baseline']['MAE_reduction_pct']:.1f}%, "
        f"RMSE -{r['test_improvement_vs_baseline']['RMSE_reduction_pct']:.1f}%.",
        f"- Mean actual demand in test period: {tm['model']['mean_actual']:.1f} rentals/hour.",
        f"- Daily totals (sum of hourly predictions): model MAE {dm['model']['MAE']:.0f}, R^2 {dm['model']['R2']:.3f}; "
        f"baseline MAE {dm['baseline']['MAE']:.0f}, R^2 {dm['baseline']['R2']:.3f}.",
        f"- Indicative 80% range (from development out-of-fold residuals) covered "
        f"{100 * r['interval_80pct_empirical_test_coverage']:.1f}% of test hours.",
        "",
        "## Residual diagnostics (test)",
        "",
        f"- Mean residual {r['test_residual_summary']['mean']:.2f}, median {r['test_residual_summary']['median']:.2f}, "
        f"5th-95th percentile [{r['test_residual_summary']['p05']:.1f}, {r['test_residual_summary']['p95']:.1f}].",
        f"- Lag-1 autocorrelation of hourly residuals: {r['test_residual_summary']['lag1_autocorrelation']:.2f} "
        "(errors persist across consecutive hours - e.g. a whole rainy afternoon - because the model uses no "
        "lagged demand).",
        "- Errors grow with demand level (heteroscedastic), as expected for count data.",
        "",
        "![Actual vs predicted](../outputs/figures/model_actual_vs_predicted.png)",
        "",
        "![Residuals](../outputs/figures/model_residuals.png)",
        "",
        "![Time series](../outputs/figures/model_time_series.png)",
        "",
        "![Permutation importance](../outputs/figures/model_permutation_importance.png)",
        "",
        "Top permutation importances (RMSE increase on test): "
        + ", ".join(f"`{k}` {v:.1f}" for k, v in list(r["permutation_importance_test"].items())[:6]) + ".",
        "",
        "## Secondary experiment: train 2011 only, test all of 2012",
        "",
        "| Model | MAE | RMSE | R^2 | Mean bias |",
        "|---|---|---|---|---|",
        f"| Baseline | {s['baseline']['MAE']:.2f} | {s['baseline']['RMSE']:.2f} | {s['baseline']['R2']:.3f} | {s['baseline']['bias']:.2f} |",
        f"| Selected config | {s['model']['MAE']:.2f} | {s['model']['RMSE']:.2f} | {s['model']['R2']:.3f} | {s['model']['bias']:.2f} |",
        "",
        "With no 2012 data in training, both models under-predict systematically (negative bias) because demand "
        "grew strongly between years and tree models cannot extrapolate a level shift. This is why the primary "
        "design uses an 18-month development window that includes early-2012 data.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
