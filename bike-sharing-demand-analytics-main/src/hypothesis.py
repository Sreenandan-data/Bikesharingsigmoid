"""Hypothesis testing: commuting-hour demand on working vs non-working days.

Pre-registered design (fixed before looking at test results):

* Commuting windows: morning hr in {7, 8} (07:00-08:59) and evening hr in {16, 17, 18}
  (16:00-18:59). Defined in ``src/config.py``.
* Unit of analysis: one calendar date. Outcome: total rentals summed over the five commuting hours.
* H0: mean commuting-window total is equal on working and non-working days.
* H1 (one-sided): mean commuting-window total is higher on working days.
* alpha = 0.05. Primary test: Welch's t-test (one-sided). Robustness: Mann-Whitney U, bootstrap CI,
  complete-window-only subset, per-year analysis and a weather/season-adjusted OLS with
  Newey-West (HAC) standard errors to account for serial correlation.

Run: ``python -m src.hypothesis``
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

from src import config
from src.features import build_features

ALPHA = 0.05


def daily_commute_totals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per date with commuting-window totals and daily covariates."""
    hours = list(config.COMMUTE_HOURS)
    sub = df[df["hr"].isin(hours)]
    commute = sub.groupby("dteday").agg(commute_total=("cnt", "sum"),
                                         commute_hours_observed=("hr", "nunique"))
    daily = df.groupby("dteday").agg(
        workingday=("workingday", "first"), holiday=("holiday", "first"),
        weekday=("weekday", "first"), season=("season", "first"), mnth=("mnth", "first"),
        yr=("yr", "first"), temp=("temp", "mean"), hum=("hum", "mean"),
        windspeed=("windspeed", "mean"), weathersit_max=("weathersit", "max"),
        daily_total=("cnt", "sum"),
    )
    # Commute-hour weather (mean over commuting hours) is a more relevant confounder.
    cw = sub.groupby("dteday").agg(commute_weathersit=("weathersit", "mean"),
                                   commute_temp=("temp", "mean"))
    out = daily.join(commute, how="left").join(cw, how="left")
    out["commute_total"] = out["commute_total"].fillna(0)
    out["commute_hours_observed"] = out["commute_hours_observed"].fillna(0).astype(int)
    out["complete_window"] = out["commute_hours_observed"] == len(hours)
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["dteday"])
    return out.sort_values("date").reset_index(drop=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a) + len(b)
    return cohens_d(a, b) * (1 - 3 / (4 * n - 9))


def welch_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Two-sided CI for mean(a) - mean(b) using the Welch-Satterthwaite df."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.5 + level / 2, dof)
    diff = a.mean() - b.mean()
    return float(diff - t * se), float(diff + t * se), float(dof)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                      seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return float(lo), float(hi)


def compare_groups(work: np.ndarray, nonwork: np.ndarray) -> dict:
    t_res = stats.ttest_ind(work, nonwork, equal_var=False, alternative="greater")
    mw = stats.mannwhitneyu(work, nonwork, alternative="greater")
    lo, hi, dof = welch_ci(work, nonwork)
    blo, bhi = bootstrap_diff_ci(work, nonwork)
    # Rank-biserial correlation for Mann-Whitney: P(X>Y) - P(X<Y).
    rbc = 2 * mw.statistic / (len(work) * len(nonwork)) - 1
    return {
        "n_working": int(len(work)), "n_nonworking": int(len(nonwork)),
        "mean_working": float(work.mean()), "mean_nonworking": float(nonwork.mean()),
        "median_working": float(np.median(work)), "median_nonworking": float(np.median(nonwork)),
        "sd_working": float(work.std(ddof=1)), "sd_nonworking": float(nonwork.std(ddof=1)),
        "mean_difference": float(work.mean() - nonwork.mean()),
        "ratio_of_means": float(work.mean() / nonwork.mean()),
        "welch_t": float(t_res.statistic), "welch_df": dof, "welch_p_one_sided": float(t_res.pvalue),
        "ci95_mean_difference_welch": [lo, hi],
        "ci95_mean_difference_bootstrap": [blo, bhi],
        "cohens_d": cohens_d(work, nonwork), "hedges_g": hedges_g(work, nonwork),
        "mann_whitney_u": float(mw.statistic), "mann_whitney_p_one_sided": float(mw.pvalue),
        "rank_biserial": float(rbc),
    }


def assumption_checks(daily: pd.DataFrame) -> dict:
    w = daily.loc[daily.workingday == 1, "commute_total"].to_numpy(float)
    nw = daily.loc[daily.workingday == 0, "commute_total"].to_numpy(float)
    sh_w, sh_nw = stats.shapiro(w), stats.shapiro(nw)
    lev = stats.levene(w, nw, center="median")
    resid = daily["commute_total"] - daily.groupby("workingday")["commute_total"].transform("mean")
    lag1 = float(pd.Series(resid.to_numpy()).autocorr(lag=1))
    return {
        "shapiro_p_working": float(sh_w.pvalue), "shapiro_p_nonworking": float(sh_nw.pvalue),
        "skew_working": float(stats.skew(w)), "skew_nonworking": float(stats.skew(nw)),
        "levene_p": float(lev.pvalue),
        "residual_lag1_autocorrelation": lag1,
        "durbin_watson": float(durbin_watson(resid.to_numpy())),
    }


def adjusted_model(daily: pd.DataFrame) -> dict:
    """OLS of log commute total on workingday with weather/season/year controls, HAC SEs."""
    d = daily.copy()
    d["log_commute"] = np.log1p(d["commute_total"])
    formula = ("log_commute ~ workingday + temp + I(temp**2) + hum + windspeed "
               "+ C(weathersit_max) + C(mnth) + yr")
    fit = smf.ols(formula, data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    coef = float(fit.params["workingday"])
    lo, hi = fit.conf_int().loc["workingday"].tolist()
    # One-sided p-value for coef > 0.
    z = float(fit.tvalues["workingday"])
    p_one = float(stats.norm.sf(z))
    lin = smf.ols(formula.replace("log_commute", "commute_total"), data=d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})
    llo, lhi = lin.conf_int().loc["workingday"].tolist()
    return {
        "formula": formula, "n": int(fit.nobs), "r2": float(fit.rsquared),
        "workingday_coef_log": coef, "workingday_ci95_log": [float(lo), float(hi)],
        "workingday_multiplier": float(np.exp(coef)),
        "workingday_multiplier_ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "workingday_z_hac": z, "workingday_p_one_sided_hac": p_one,
        "linear_workingday_coef": float(lin.params["workingday"]),
        "linear_workingday_ci95": [float(llo), float(lhi)],
        "linear_workingday_p_two_sided_hac": float(lin.pvalues["workingday"]),
    }


def secondary_weather_test(df: pd.DataFrame) -> dict:
    """H2: daily rentals are lower on days whose worst weather is light rain/snow or worse.

    Compared within the same season-year cells to limit seasonal confounding (stratified
    Mann-Whitney via the van Elteren-like approach is approximated here by reporting the
    unstratified test and the mean of within-stratum differences)."""
    daily = daily_commute_totals(df)
    bad = daily["weathersit_max"] >= 3
    a = daily.loc[~bad, "daily_total"].to_numpy(float)
    b = daily.loc[bad, "daily_total"].to_numpy(float)
    mw = stats.mannwhitneyu(a, b, alternative="greater")
    t_res = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    strata = []
    for (s, y), g in daily.groupby(["season", "yr"]):
        gb = g["weathersit_max"] >= 3
        if gb.sum() >= 3 and (~gb).sum() >= 3:
            strata.append(g.loc[~gb, "daily_total"].mean() - g.loc[gb, "daily_total"].mean())
    return {
        "definition": "Rainy day = at least one hour with weathersit >= 3 (light rain/snow or worse).",
        "n_not_rainy": int(len(a)), "n_rainy": int(len(b)),
        "mean_not_rainy": float(a.mean()), "mean_rainy": float(b.mean()),
        "median_not_rainy": float(np.median(a)), "median_rainy": float(np.median(b)),
        "welch_t": float(t_res.statistic), "welch_p_one_sided": float(t_res.pvalue),
        "mann_whitney_p_one_sided": float(mw.pvalue),
        "mean_within_season_year_difference": float(np.mean(strata)) if strata else None,
        "n_strata": len(strata),
    }


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        from src.data_validation import load_raw
        df = build_features(load_raw(config.HOUR_CSV))
    daily = daily_commute_totals(df)

    def groups(d):
        return (d.loc[d.workingday == 1, "commute_total"].to_numpy(float),
                d.loc[d.workingday == 0, "commute_total"].to_numpy(float))

    primary = compare_groups(*groups(daily))
    complete = daily[daily.complete_window]
    sensitivity = {
        "complete_windows_only": compare_groups(*groups(complete)),
        "year_2011": compare_groups(*groups(daily[daily.yr == 0])),
        "year_2012": compare_groups(*groups(daily[daily.yr == 1])),
        "excluding_holidays_from_nonworking": compare_groups(
            *groups(daily[~((daily.workingday == 0) & (daily.holiday == 1))])),
    }
    hourly_profile = (df.groupby(["workingday", "hr"])["cnt"].mean().unstack(0)
                      .rename(columns={0: "nonworking", 1: "working"}))
    results = {
        "research_question": "Is bike demand during commuting hours higher on working days than on "
                             "non-working days?",
        "H0": "The mean daily rental total across the commuting windows is the same for working and "
              "non-working days.",
        "H1": "The mean daily rental total across the commuting windows is higher on working days.",
        "alpha": ALPHA,
        "commute_windows": {"morning_hours": list(config.MORNING_COMMUTE_HOURS),
                            "evening_hours": list(config.EVENING_COMMUTE_HOURS),
                            "description": "07:00-08:59 and 16:00-18:59 (hr values 7-8 and 16-18)"},
        "n_days": int(len(daily)), "n_days_incomplete_window": int((~daily.complete_window).sum()),
        "n_holiday_dates": int(daily["holiday"].sum()),
        "n_missing_hours_total": int(len(daily) * 24 - len(df)),
        "primary": primary,
        "reject_H0": bool(primary["welch_p_one_sided"] < ALPHA),
        "assumptions": assumption_checks(daily),
        "sensitivity": sensitivity,
        "adjusted_ols_hac": adjusted_model(daily),
        "secondary_weather_hypothesis": secondary_weather_test(df),
        "hourly_profile": {str(k): v for k, v in hourly_profile.round(1).to_dict().items()},
    }
    results["all_checks_significant"] = bool(
        primary["welch_p_one_sided"] < ALPHA and primary["mann_whitney_p_one_sided"] < ALPHA
        and all(v["welch_p_one_sided"] < ALPHA for v in sensitivity.values())
        and results["adjusted_ols_hac"]["workingday_p_one_sided_hac"] < ALPHA)
    return results


def make_figure(df: pd.DataFrame, results: dict, path) -> None:
    daily = daily_commute_totals(df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    labels = {1: "Working day", 0: "Non-working day"}
    data = [daily.loc[daily.workingday == k, "commute_total"] for k in (1, 0)]
    axes[0].boxplot(data, tick_labels=[labels[1], labels[0]], showmeans=True)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        axes[0].scatter(rng.normal(i, 0.05, len(d)), d, s=6, alpha=0.3)
    axes[0].set_title("Daily commuting-window rentals\n(07-09h + 16-19h)")
    axes[0].set_ylabel("Rentals per day in commuting windows")

    p = results["primary"]
    axes[1].bar(["Working", "Non-working"], [p["mean_working"], p["mean_nonworking"]],
                color=["#1f77b4", "#ff7f0e"])
    lo, hi = p["ci95_mean_difference_welch"]
    axes[1].set_title(f"Group means\ndiff = {p['mean_difference']:.0f} (95% CI {lo:.0f} to {hi:.0f})\n"
                      f"Welch p = {p['welch_p_one_sided']:.2g}, Hedges g = {p['hedges_g']:.2f}")
    axes[1].set_ylabel("Mean commuting-window rentals")

    prof = df.groupby(["workingday", "hr"])["cnt"].mean().unstack(0)
    for hrs in (config.MORNING_COMMUTE_HOURS, config.EVENING_COMMUTE_HOURS):
        axes[2].axvspan(min(hrs) - 0.5, max(hrs) + 0.5, color="grey", alpha=0.15)
    axes[2].plot(prof.index, prof[1], marker="o", label=labels[1])
    axes[2].plot(prof.index, prof[0], marker="o", label=labels[0])
    axes[2].set_xticks(range(0, 24, 2))
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Mean rentals per hour")
    axes[2].set_title("Hourly profile (shaded = commuting windows)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt_p(p: float) -> str:
    return f"{p:.3g}" if p >= 1e-300 else "< 1e-300"


def to_markdown(r: dict) -> str:
    p, a, adj, s = r["primary"], r["assumptions"], r["adjusted_ols_hac"], r["sensitivity"]
    w2 = r["secondary_weather_hypothesis"]
    lines = [
        "# Hypothesis Test Results",
        "",
        "Generated by `python -m src.hypothesis`. All numbers below are computed from `data/raw/hour.csv`.",
        "",
        "## Design (fixed before testing)",
        "",
        f"- **Research question:** {r['research_question']}",
        f"- **H0:** {r['H0']}",
        f"- **H1:** {r['H1']}",
        f"- **Commuting windows:** {r['commute_windows']['description']}.",
        "- **Unit of analysis:** calendar date; outcome = sum of `cnt` over the five commuting hours.",
        f"- **Significance level:** alpha = {r['alpha']} (one-sided).",
        "- **Primary test:** Welch's independent-samples t-test (unequal variances).",
        "- **Robustness:** Mann-Whitney U, bootstrap CI (10,000 resamples), complete-window days only, "
        "per-year tests, excluding holidays, and weather/season-adjusted OLS with Newey-West (HAC, 7 lags) SEs.",
        "",
        "## Primary result",
        "",
        "| Statistic | Working days | Non-working days |",
        "|---|---|---|",
        f"| n (days) | {p['n_working']} | {p['n_nonworking']} |",
        f"| Mean | {p['mean_working']:.1f} | {p['mean_nonworking']:.1f} |",
        f"| Median | {p['median_working']:.1f} | {p['median_nonworking']:.1f} |",
        f"| SD | {p['sd_working']:.1f} | {p['sd_nonworking']:.1f} |",
        "",
        f"- Difference in means (working - non-working): **{p['mean_difference']:.1f} rentals/day** "
        f"(ratio {p['ratio_of_means']:.2f}x)",
        f"- 95% CI (Welch): [{p['ci95_mean_difference_welch'][0]:.1f}, {p['ci95_mean_difference_welch'][1]:.1f}]; "
        f"bootstrap 95% CI: [{p['ci95_mean_difference_bootstrap'][0]:.1f}, {p['ci95_mean_difference_bootstrap'][1]:.1f}]",
        f"- Welch t = {p['welch_t']:.2f}, df = {p['welch_df']:.1f}, one-sided p = {_fmt_p(p['welch_p_one_sided'])}",
        f"- Effect size: Cohen's d = {p['cohens_d']:.2f}, Hedges' g = {p['hedges_g']:.2f}",
        f"- Mann-Whitney U = {p['mann_whitney_u']:.0f}, one-sided p = {_fmt_p(p['mann_whitney_p_one_sided'])}, "
        f"rank-biserial r = {p['rank_biserial']:.2f}",
        f"- **Decision:** {'Reject H0' if r['reject_H0'] else 'Fail to reject H0'} at alpha = {r['alpha']}.",
        "",
        "![Hypothesis figure](../outputs/figures/hypothesis_commute.png)",
        "",
        "## Assumption checks",
        "",
        f"- Normality (Shapiro-Wilk): p = {a['shapiro_p_working']:.3g} (working), "
        f"{a['shapiro_p_nonworking']:.3g} (non-working); skewness {a['skew_working']:.2f} / {a['skew_nonworking']:.2f}. "
        "With n > 200 per group the CLT makes the t-test on means reasonably robust; the Mann-Whitney test "
        "and bootstrap CI do not assume normality.",
        f"- Equal variances (Levene, median-centred): p = {a['levene_p']:.3g}. Welch's test does not assume equal variances.",
        f"- Independence: lag-1 autocorrelation of group-demeaned daily totals = {a['residual_lag1_autocorrelation']:.2f} "
        f"(Durbin-Watson {a['durbin_watson']:.2f}). Days are serially dependent (trend, seasons, weather spells), "
        "so the simple tests overstate the effective sample size. The HAC-adjusted regression below addresses this.",
        "",
        "## Sensitivity analyses",
        "",
        "| Analysis | n work / non-work | Mean diff | 95% CI (Welch) | Welch p (1-sided) | Hedges g |",
        "|---|---|---|---|---|---|",
    ]
    names = {"complete_windows_only": "Complete commuting windows only", "year_2011": "2011 only",
             "year_2012": "2012 only", "excluding_holidays_from_nonworking": "Weekends only (holidays excluded)"}
    for k, v in s.items():
        lines.append(f"| {names[k]} | {v['n_working']} / {v['n_nonworking']} | {v['mean_difference']:.1f} | "
                     f"[{v['ci95_mean_difference_welch'][0]:.1f}, {v['ci95_mean_difference_welch'][1]:.1f}] | "
                     f"{_fmt_p(v['welch_p_one_sided'])} | {v['hedges_g']:.2f} |")
    lines += [
        "",
        "### Weather- and season-adjusted model (confounding + serial dependence)",
        "",
        f"`{adj['formula']}` estimated by OLS with Newey-West HAC standard errors (7 lags), n = {adj['n']}, "
        f"R^2 = {adj['r2']:.3f}.",
        "",
        f"- Working-day multiplier on commuting-window rentals: **{adj['workingday_multiplier']:.2f}x** "
        f"(95% CI {adj['workingday_multiplier_ci95'][0]:.2f}-{adj['workingday_multiplier_ci95'][1]:.2f}), "
        f"HAC z = {adj['workingday_z_hac']:.2f}, one-sided p = {_fmt_p(adj['workingday_p_one_sided_hac'])}.",
        f"- Linear-scale adjusted difference: {adj['linear_workingday_coef']:.1f} rentals/day "
        f"(95% CI {adj['linear_workingday_ci95'][0]:.1f} to {adj['linear_workingday_ci95'][1]:.1f}).",
        "",
        "## Secondary hypothesis (weather)",
        "",
        f"H0: mean daily rentals are equal on rainy and non-rainy days; H1: lower on rainy days. {w2['definition']}",
        "",
        f"- Non-rainy days: n = {w2['n_not_rainy']}, mean {w2['mean_not_rainy']:.0f}, median {w2['median_not_rainy']:.0f}",
        f"- Rainy days: n = {w2['n_rainy']}, mean {w2['mean_rainy']:.0f}, median {w2['median_rainy']:.0f}",
        f"- Welch one-sided p = {_fmt_p(w2['welch_p_one_sided'])}; Mann-Whitney one-sided p = {_fmt_p(w2['mann_whitney_p_one_sided'])}",
        f"- Mean within season-year difference ({w2['n_strata']} strata): "
        f"{w2['mean_within_season_year_difference']:.0f} rentals/day",
        "",
        "## Conclusion (plain English)",
        "",
        (f"On working days, the five commuting hours carry on average about {p['mean_difference']:.0f} more rentals "
         f"per day than on weekends and holidays ({p['mean_working']:.0f} vs {p['mean_nonworking']:.0f}, "
         f"about {p['ratio_of_means']:.1f}x). "
         + ("The difference is statistically significant under every test and sensitivity check above, "
            "including after adjusting for weather, month and year with serial-correlation-robust standard errors. "
            if r["all_checks_significant"] else
            "Note: not every sensitivity check reached significance; see the tables above. ")
         + f"The standardised effect is large (Hedges' g {p['hedges_g']:.2f})." if r["reject_H0"] else
         "The data do not provide sufficient evidence that commuting-hour demand is higher on working days."),
        "",
        "## Limitations",
        "",
        "- Observational data: this is an association, not a causal effect of 'working day' status. Working days "
        "differ from non-working days in many ways (commuter mix, events, tourism).",
        "- Commuting windows are a design choice; results could differ for other windows. They were fixed in "
        "`src/config.py` before running the test.",
        f"- Holidays are few ({r['n_holiday_dates']} dates) and grouped with weekends.",
        "- Missing hourly records ({m} absent hours across the series; {i} dates have an incomplete commuting window) "
        "are treated as zero rentals in the window sum; the complete-window sensitivity analysis checks this.".format(
            m=r["n_missing_hours_total"], i=r["n_days_incomplete_window"]),
        "- System growth between 2011 and 2012 inflates variance; per-year tests and the `yr` control address this.",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    from src.data_validation import load_raw
    config.ensure_dirs()
    df = build_features(load_raw(config.HOUR_CSV))
    results = run(df)
    make_figure(df, results, config.FIGURES_DIR / "hypothesis_commute.png")
    (config.OUTPUTS_DIR / "hypothesis_results.json").write_text(json.dumps(results, indent=2))
    (config.REPORTS_DIR / "hypothesis_results.md").write_text(to_markdown(results), encoding="utf-8")
    p = results["primary"]
    print(f"Working mean {p['mean_working']:.1f} vs non-working {p['mean_nonworking']:.1f}; "
          f"Welch p={p['welch_p_one_sided']:.3g}; reject H0={results['reject_H0']}")
    return results


if __name__ == "__main__":
    main()
