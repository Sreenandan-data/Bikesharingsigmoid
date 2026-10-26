"""Deterministic natural-language query layer for the operations dashboard.

Design principles
-----------------
* Intent detection and entity extraction are rule-based (regular expressions), so the app works
  offline and without any API key.
* Every number in an answer is computed from the dataset (``src.analysis``), the stored hypothesis
  results, or the trained model (``src.predict``). The language layer only selects and formats.
* If a question cannot be mapped to a supported intent the engine says so and lists examples; it
  never guesses an answer.
* Optional: if ``BIKE_LLM_API_KEY`` (and optionally ``BIKE_LLM_BASE_URL`` / ``BIKE_LLM_MODEL``) is
  set, an OpenAI-compatible chat endpoint may be asked to map an *unrecognised* question onto one of
  the supported intent names. The LLM never produces numbers; the deterministic handler still
  computes the answer.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

from src import analysis
from src.features import WEATHER_LABELS

SUPPORTED_EXAMPLES = [
    "When is bike demand highest on working days?",
    "Compare demand on working days and non-working days.",
    "How does weather affect rental demand?",
    "What is the average demand in summer?",
    "Estimate bike rentals at 8 AM in summer on a working day.",
    "Compare expected demand during clear weather and light rain.",
    "Which hours should the operations team monitor most closely?",
    "Was the working-day difference statistically significant?",
    "How accurate is the demand model?",
    "Which days had unusual demand?",
    "How did demand grow from 2011 to 2012?",
]

INTENTS = ["help", "anomalies", "predict", "model_performance", "hypothesis", "monitor_hours",
           "peak_hours", "compare_day_types", "weather_effect", "season_average", "trend",
           "hourly_pattern", "overview"]

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}
MONTH_ABBR = {m[:3]: i for m, i in MONTHS.items() if m != "may"}


@dataclass
class Entities:
    hours: list[int] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    day_types: list[int] = field(default_factory=list)   # 1 = working day, 0 = non-working day
    weathers: list[int] = field(default_factory=list)
    month: int | None = None
    weekday: int | None = None
    holiday: bool = False
    year: int | None = None
    temp_c: float | None = None
    hum_pct: float | None = None
    wind_kmh: float | None = None


@dataclass
class Answer:
    intent: str
    text: str
    table: pd.DataFrame | None = None
    chart: dict | None = None        # {"kind": "line"|"bar", "data": DataFrame, "x": str, "y": [..]}
    entities: Entities | None = None
    understood: bool = True


# ----------------------------------------------------------------------------------------------
# Entity extraction
# ----------------------------------------------------------------------------------------------
def _consume(pattern: str, text: str):
    """Return (matches, text with matches blanked) so later patterns don't re-match."""
    matches = list(re.finditer(pattern, text))
    for m in reversed(matches):
        text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    return matches, text


def extract_entities(question: str) -> Entities:
    t = " " + question.lower().replace("–", "-") + " "
    e = Entities()

    # Numeric weather quantities first, so "25 degrees" is not read as an hour.
    ms, t = _consume(r"(-?\d{1,2}(?:\.\d+)?)\s*(?:°\s*c|degrees?\s*(?:c|celsius)?|deg\b|celsius|c\b)", t)
    if ms:
        e.temp_c = float(ms[0].group(1))
    ms, t = _consume(r"humidity\s*(?:of|at|=|is)?\s*(\d{1,3})\s*%?|(\d{1,3})\s*%\s*humidity", t)
    if ms:
        e.hum_pct = float(ms[0].group(1) or ms[0].group(2))
    ms, t = _consume(r"(\d{1,3}(?:\.\d+)?)\s*(?:km/?h|kph|kmh)", t)
    if ms:
        e.wind_kmh = float(ms[0].group(1))

    ms, t = _consume(r"\b(2011|2012)\b", t)
    if ms:
        e.year = int(ms[0].group(1))

    # Hours.
    found: list[tuple[int, int]] = []  # (position, hour)
    ms, t = _consume(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", t)
    for m in ms:
        h = int(m.group(1)) % 12
        if m.group(3).startswith("p"):
            h += 12
        if 0 <= h <= 23:
            found.append((m.start(), h))
    ms, t = _consume(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
    found += [(m.start(), int(m.group(1))) for m in ms]
    ms, t = _consume(r"\b(?:hour|hr)\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:h|hrs|o'?clock)\b", t)
    found += [(m.start(), int(m.group(1) or m.group(2))) for m in ms if int(m.group(1) or m.group(2)) < 24]
    for word, h in (("noon", 12), ("midday", 12), ("midnight", 0)):
        ms, t = _consume(rf"\b{word}\b", t)
        found += [(m.start(), h) for m in ms]
    for pat, h in ((r"morning (?:rush|commute|peak)", 8), (r"evening (?:rush|commute|peak)", 17)):
        ms, t = _consume(pat, t)
        found += [(m.start(), h) for m in ms]
    e.hours = [h for _, h in sorted(found)]

    # Weather (most specific first).
    wfound: list[tuple[int, int]] = []
    for pat, code in ((r"heavy (?:rain|snow)|thunderstorms?|storms?|blizzard|ice pellets|hail", 4),
                      (r"light (?:rain|snow)|drizzle|showers?|rain(?:y|ing)?|snow(?:y|ing)?|wet weather", 3),
                      (r"mist(?:y)?|cloudy|overcast|fog(?:gy)?|cloud(?:s)?", 2),
                      (r"clear(?: weather| skies?)?|sunny|fair weather|good weather|nice weather|dry weather", 1)):
        ms, t = _consume(rf"\b(?:{pat})\b", t)
        wfound += [(m.start(), code) for m in ms]
    seen = []
    for _, c in sorted(wfound):
        if c not in seen:
            seen.append(c)
    e.weathers = seen

    # Day type.
    dfound: list[tuple[int, int]] = []
    ms, t = _consume(r"\bholidays?\b", t)
    if ms:
        e.holiday = True
        dfound += [(m.start(), 0) for m in ms]
    ms, t = _consume(r"\bnon[- ]?work(?:ing)?[- ]?days?\b|\bweekends?\b|\bdays? off\b|\bsaturdays?\b|\bsundays?\b"
                     r"|\bnon[- ]?working\b", t)
    dfound += [(m.start(), 0) for m in ms]
    for name, wd in (("saturday", 6), ("sunday", 0)):
        if re.search(rf"\b{name}s?\b", question.lower()):
            e.weekday = wd
    ms2 = list(re.finditer(r"\b(monday|tuesday|wednesday|thursday|friday)s?\b", t))
    if ms2:
        e.weekday = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday"].index(ms2[0].group(1))
    ms, t = _consume(r"\bwork(?:ing)?[- ]?days?\b|\bweekdays?\b|\bbusiness days?\b|\bworkdays?\b"
                     r"|\b(?:monday|tuesday|wednesday|thursday|friday)s?\b|\bcommut\w*", t)
    dfound += [(m.start(), 1) for m in ms]
    seen = []
    for _, c in sorted(dfound):
        if c not in seen:
            seen.append(c)
    e.day_types = seen

    # Seasons and months.
    sfound = []
    for pat, name in ((r"spring", "spring"), (r"summer", "summer"), (r"fall|autumn", "fall"), (r"winter", "winter")):
        sfound += [(m.start(), name) for m in re.finditer(rf"\b(?:{pat})\b", t)]
    e.seasons = list(dict.fromkeys(n for _, n in sorted(sfound)))
    for name, i in MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            e.month = i
            break
    else:
        for ab, i in MONTH_ABBR.items():
            if re.search(rf"\b{ab}\b", t):
                e.month = i
                break
    return e


# ----------------------------------------------------------------------------------------------
# Intent detection
# ----------------------------------------------------------------------------------------------
_PATTERNS = [
    ("help", r"\b(help|what can (?:you|i) ask|examples?|supported questions?)\b"),
    ("anomalies", r"\b(anomal\w*|unusual|outliers?|abnormal|hurricane|sandy|irene|strange|odd days?|events?)\b"),
    ("predict", r"\b(estimate\w*|predict\w*|forecast\w*|expected|expect|projection|what if|how many (?:bikes|rentals|rides)"
                r"(?: will| would| should| can| do we)|scenario)\b"),
    ("model_performance", r"\b(accura\w*|model performance|performance of the model|how good|reliab\w*|mae|rmse|r2|r\^2|"
                          r"r-squared|error rate|how well does the model)\b"),
    ("hypothesis", r"\b(hypothes\w*|significan\w*|statistical\w*|t-test|p-value|p value|welch|mann-whitney|null)\b"),
    ("monitor_hours", r"\b(monitor\w*|watch|closely|attention|critical hours?|staff\w*|rebalanc\w*|focus on|prioriti\w*)\b"),
    ("peak_hours", r"\b(highest|peak\w*|busiest|maximum|most (?:demand|rentals|busy)|top hours?|rush hours?|when is .* (?:high|busy))\b"),
    ("compare_day_types", r"\b(compar\w*|versus|vs\.?|difference|differ)\b"),
    ("weather_effect", r"\b(weather|rain\w*|snow\w*|mist\w*|clear|sunny|temperature|temp|hot|cold|humid\w*|wind\w*|cloud\w*)\b"),
    ("season_average", r"\b(seasons?|spring|summer|fall|autumn|winter|seasonal\w*)\b"),
    ("trend", r"\b(trend|growth|grow\w*|increase\w*|2011|2012|year over year|yoy|monthly|by month|over time)\b"),
    ("hourly_pattern", r"\b(hourly|by hour|time of day|daily pattern|hour of day|pattern)\b"),
    ("overview", r"\b(overview|summary|summari[sz]e|total|average demand|mean demand|typical demand|how many rentals)\b"),
]


def detect_intent(question: str, e: Entities | None = None) -> str:
    q = question.lower()
    e = e or extract_entities(question)
    for name, pat in _PATTERNS:
        if re.search(pat, q):
            if name == "compare_day_types":
                if len(e.day_types) >= 2 or re.search(r"working|weekend|weekday|holiday", q):
                    return name
                if len(e.weathers) >= 2 or "weather" in q:
                    return "weather_effect"
                if len(e.seasons) >= 2:
                    return "season_average"
                if e.year or re.search(r"\byears?\b", q):
                    return "trend"
                continue
            if name == "peak_hours" and not re.search(r"\b(hour|time|when)\b", q) and e.seasons:
                return "season_average"
            return name
    # Bare scenario descriptions such as "8 am summer working day clear weather".
    if e.hours and (e.seasons or e.weathers or e.day_types):
        return "predict"
    return "unknown"


def llm_intent(question: str) -> str | None:
    """Optional LLM fallback: map an unrecognised question to a supported intent label."""
    key = os.environ.get("BIKE_LLM_API_KEY")
    if not key:
        return None
    base = os.environ.get("BIKE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("BIKE_LLM_MODEL", "gpt-4o-mini")
    prompt = ("Classify the user's question about a bike-sharing demand dashboard into exactly one of "
              f"these labels: {', '.join(INTENTS + ['unknown'])}. Reply with the label only.\n\nQuestion: {question}")
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            label = json.loads(resp.read())["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return None
    return label if label in INTENTS else None


# ----------------------------------------------------------------------------------------------
# Answering
# ----------------------------------------------------------------------------------------------
@dataclass
class Context:
    df: pd.DataFrame
    bundle: dict | None = None
    hypothesis: dict | None = None
    model_metrics: dict | None = None
    anomalies: pd.DataFrame | None = None
    season_scheme: str = "official"
    buffer_pct: float = 15.0
    use_llm: bool = True


def _hr(h: int) -> str:
    return f"{h:02d}:00"


def _season_note(scheme: str) -> str:
    if scheme == "official":
        return ("_Season labels follow the official dataset README (code 2 = 'summer'). In the data these codes "
                "switch on astronomical dates, so 'summer' here covers ~21 Mar-20 Jun; switch the season "
                "interpretation in the sidebar to use calendar labels._")
    return ("_Season labels use the calendar interpretation (code 3 = summer, ~21 Jun-22 Sep), not the "
            "README's literal labels._")


def help_answer(understood: bool = True, intent: str = "help") -> Answer:
    head = ("I can answer these kinds of questions from the dataset and the trained model:" if understood else
            "Sorry - I couldn't match that question to an analysis I can compute, so I won't guess. "
            "Try rephrasing it like one of these:")
    return Answer(intent, head + "\n\n" + "\n".join(f"- {q}" for q in SUPPORTED_EXAMPLES), understood=understood)


def answer_peak(ctx: Context, e: Entities) -> Answer:
    wd = e.day_types[0] if len(e.day_types) == 1 else None
    season = analysis.season_code(e.seasons[0], ctx.season_scheme) if e.seasons else None
    tab = analysis.peak_hours(ctx.df, workingday=wd, top=5, season=season)
    scope = {1: "working days", 0: "non-working days", None: "all days"}[wd]
    if season:
        scope += f" in {e.seasons[0]}"
    top = tab.iloc[0]
    text = (f"On **{scope}**, demand is highest at **{_hr(int(top.hr))}** with an average of "
            f"**{top['mean']:.0f} rentals/hour** (90th percentile {top.p90:.0f}). "
            f"Next busiest: " + ", ".join(f"{_hr(int(r.hr))} ({r.mean:.0f})" for r in tab.iloc[1:].itertuples()) + ".")
    prof = ctx.df if wd is None else ctx.df[ctx.df.workingday == wd]
    if season:
        prof = prof[prof.season == season]
    chart = {"kind": "bar", "data": prof.groupby("hr")["cnt"].mean().reset_index(), "x": "hr", "y": ["cnt"],
             "title": f"Mean rentals by hour - {scope}"}
    if season:
        text += "\n\n" + _season_note(ctx.season_scheme)
    out = tab.rename(columns={"hr": "hour", "mean": "mean/hour", "median": "median/hour", "p90": "90th pct", "n": "hours observed"})
    return Answer("peak_hours", text, out.round(1), chart, e)


def answer_compare_day_types(ctx: Context, e: Entities) -> Answer:
    tab = analysis.compare_day_types(ctx.df)
    w, n = tab.iloc[0], tab.iloc[1]
    text = (f"**Working days** average **{w['mean rentals/day']:.0f} rentals/day** vs **{n['mean rentals/day']:.0f}** on "
            f"**non-working days** ({w['mean rentals/day'] / n['mean rentals/day']:.2f}x). The shape differs more than the total: "
            f"working days peak at {_hr(int(w['peak hour']))} ({w['peak-hour mean']:.0f}/hour) with a second commute peak, while "
            f"non-working days peak at {_hr(int(n['peak hour']))} ({n['peak-hour mean']:.0f}/hour) with a broad midday plateau. "
            f"During the commuting hours (07-09h, 16-19h) working days average {w['commute-hours mean/hour']:.0f}/hour vs "
            f"{n['commute-hours mean/hour']:.0f}/hour. Registered users make up {w['registered share %']:.0f}% of working-day rentals "
            f"vs {n['registered share %']:.0f}% on non-working days.")
    if ctx.hypothesis:
        p = ctx.hypothesis["primary"]
        text += (f"\n\nThe pre-registered test of commuting-window demand found a difference of {p['mean_difference']:.0f} "
                 f"rentals/day (95% CI {p['ci95_mean_difference_welch'][0]:.0f}-{p['ci95_mean_difference_welch'][1]:.0f}, "
                 f"Welch one-sided p = {p['welch_p_one_sided']:.2g}).")
    prof = analysis.hourly_profile(ctx.df, by="day_type")
    chart = {"kind": "line", "data": prof, "x": "hr", "y": ["working day", "non-working day"],
             "title": "Mean rentals by hour: working vs non-working days"}
    return Answer("compare_day_types", text, tab.round(1), chart, e)


def answer_weather(ctx: Context, e: Entities) -> Answer:
    tab = analysis.weather_summary(ctx.df)
    temp = analysis.temperature_summary(ctx.df)
    r = {int(x.weathersit): x for x in tab.itertuples()}
    text = ("Weather has a strong effect. Comparing each hour with the average for the same hour, working-day status, "
            "month and year (so time-of-day and seasonal mix are controlled):\n\n")
    for code in sorted(r):
        x = r[code]
        text += (f"- **{x.weather}**: {x.mean_hourly:.0f} rentals/hour on average over {x.hours:,} hours; "
                 f"{x.adjusted_change_pct:+.0f}% vs. comparable hours\n")
    if 4 in r and r[4].hours < 30:
        text += f"\n_Heavy rain/snow occurs in only {r[4].hours} hours, so its estimate is very uncertain._\n"
    best = temp.loc[temp["mean_hourly"].idxmax()]
    text += (f"\nTemperature matters too: mean demand rises from {temp.iloc[0].mean_hourly:.0f}/hour below 5 °C to "
             f"{best.mean_hourly:.0f}/hour at {best.temp_band} °C (raw averages, not adjusted for season).")
    if ctx.hypothesis:
        w = ctx.hypothesis["secondary_weather_hypothesis"]
        text += (f"\n\nDays with any light rain/snow averaged {w['mean_rainy']:.0f} rentals vs {w['mean_not_rainy']:.0f} "
                 f"on other days (Mann-Whitney one-sided p = {w['mann_whitney_p_one_sided']:.2g}). This is an association, "
                 "not proof of causation.")
    chart = {"kind": "bar", "data": temp.astype({"temp_band": str}), "x": "temp_band", "y": ["mean_hourly"],
             "title": "Mean rentals/hour by temperature band (°C)"}
    return Answer("weather_effect", text, tab.round(1), chart, e)


def answer_season(ctx: Context, e: Entities) -> Answer:
    tab = analysis.season_summary(ctx.df, ctx.season_scheme)
    if e.seasons:
        rows = tab[tab["season_name"].isin(e.seasons)]
        parts = [f"In **{r.season_name}** ({r.date_range}), average demand is **{r.mean_daily:.0f} rentals/day** "
                 f"(median {r.median_daily:.0f}; {r.mean_hourly:.0f}/hour over {r.days} days)." for r in rows.itertuples()]
        text = " ".join(parts)
        if ctx.season_scheme == "official" and "summer" in e.seasons:
            cal = tab[tab["season"] == analysis.season_code("summer", "calendar")].iloc[0]
            text += (f" If you mean calendar summer (~21 Jun-22 Sep, dataset code 3), the average is "
                     f"{cal.mean_daily:.0f} rentals/day.")
    else:
        top = tab.loc[tab["mean_daily"].idxmax()]
        low = tab.loc[tab["mean_daily"].idxmin()]
        text = (f"Demand is highest in **{top.season_name}** ({top.mean_daily:.0f}/day) and lowest in "
                f"**{low.season_name}** ({low.mean_daily:.0f}/day).")
    text += "\n\n" + _season_note(ctx.season_scheme)
    chart = {"kind": "bar", "data": tab, "x": "season_name", "y": ["mean_daily"], "title": "Mean rentals/day by season"}
    return Answer("season_average", text, tab.round(1), chart, e)


def answer_monitor(ctx: Context, e: Entities) -> Answer:
    tab = analysis.monitor_hours(ctx.df, top=6)
    lines = [f"- **{r.day_type}, {_hr(int(r.hr))}**: mean {r.mean:.0f}/hour, busy-day level (90th pct) {r.p90:.0f}, "
             f"variability (CV) {r.cv:.2f}" for r in tab.itertuples()]
    text = ("Ranked by how high demand gets on a busy day (90th percentile per hour slot), these slots need the "
            "closest monitoring:\n\n" + "\n".join(lines) +
            "\n\nOperationally: working-day 08:00 and 17:00-18:00 are the clear pressure points (fleet rebalancing "
            "before the morning peak and between peaks). Non-working-day early afternoons are lower but more variable, "
            "so they depend more on weather. This ranking uses system-wide counts; station-level hotspots cannot be "
            "identified from this dataset.")
    return Answer("monitor_hours", text, tab.round(2), None, e)


def answer_hypothesis(ctx: Context, e: Entities) -> Answer:
    if not ctx.hypothesis:
        return Answer("hypothesis", "Hypothesis results are not available. Run `python -m src.hypothesis`.", understood=True)
    h = ctx.hypothesis
    p, adj = h["primary"], h["adjusted_ols_hac"]
    text = (f"**H0:** {h['H0']}\n\n**H1:** {h['H1']}\n\n"
            f"Commuting windows: {h['commute_windows']['description']}; alpha = {h['alpha']}.\n\n"
            f"- Working days (n={p['n_working']}): mean {p['mean_working']:.0f}, median {p['median_working']:.0f}\n"
            f"- Non-working days (n={p['n_nonworking']}): mean {p['mean_nonworking']:.0f}, median {p['median_nonworking']:.0f}\n"
            f"- Difference {p['mean_difference']:.0f} (95% CI {p['ci95_mean_difference_welch'][0]:.0f} to "
            f"{p['ci95_mean_difference_welch'][1]:.0f}); Welch t = {p['welch_t']:.2f}, one-sided p = {p['welch_p_one_sided']:.2g}; "
            f"Hedges' g = {p['hedges_g']:.2f}\n"
            f"- Mann-Whitney one-sided p = {p['mann_whitney_p_one_sided']:.2g}\n"
            f"- Weather/season-adjusted with HAC errors: working days have {adj['workingday_multiplier']:.2f}x the commuting "
            f"demand (95% CI {adj['workingday_multiplier_ci95'][0]:.2f}-{adj['workingday_multiplier_ci95'][1]:.2f})\n\n"
            f"**Decision:** {'reject H0' if h['reject_H0'] else 'fail to reject H0'}. This is an observational association, "
            "not a causal effect.")
    return Answer("hypothesis", text, None, None, e)


def answer_model_perf(ctx: Context, e: Entities) -> Answer:
    if not ctx.model_metrics:
        return Answer("model_performance", "Model metrics are not available. Run `python -m src.train_model`.")
    m = ctx.model_metrics
    tm = m["test_metrics_hourly"]
    text = (f"On the held-out test period ({m['design']['test_period'][0][:10]} to {m['design']['test_period'][1][:10]}, "
            f"never used for model selection), the selected **{m['selected']['family']}** model has "
            f"MAE **{tm['model']['MAE']:.1f}**, RMSE **{tm['model']['RMSE']:.1f}** and R² **{tm['model']['R2']:.3f}** "
            f"(rentals/hour). The hour x working-day baseline scores MAE {tm['baseline']['MAE']:.1f}, RMSE "
            f"{tm['baseline']['RMSE']:.1f}, R² {tm['baseline']['R2']:.3f}; i.e. the model cuts MAE by "
            f"{m['test_improvement_vs_baseline']['MAE_reduction_pct']:.0f}%. Average demand in that period was "
            f"{tm['model']['mean_actual']:.0f} rentals/hour.")
    tab = pd.DataFrame([{"model": k, **{x: round(v[x], 3) for x in ("MAE", "RMSE", "R2", "bias")}} for k, v in tm.items()])
    return Answer("model_performance", text, tab, None, e)


def answer_anomalies(ctx: Context, e: Entities) -> Answer:
    if ctx.anomalies is None or ctx.anomalies.empty:
        return Answer("anomalies", "Anomaly results are not available. Run `python -m src.anomaly`.")
    a = ctx.anomalies.copy()
    low = a[a["direction"] == "low"].head(5)
    high = a[a["direction"] == "high"].head(3)
    text = (f"{len(a)} days were flagged where observed rentals differ strongly from the model's expectation for that "
            "date's calendar and weather (cross-fitted, robust z-score threshold). Largest shortfalls: "
            + ", ".join(f"{str(r.date)[:10]} ({r.actual:.0f} vs ~{r.expected:.0f} expected)" for r in low.itertuples()) + ".")
    if len(high):
        text += " Largest surpluses: " + ", ".join(f"{str(r.date)[:10]} ({r.actual:.0f} vs ~{r.expected:.0f})" for r in high.itertuples()) + "."
    text += (" Flags are candidates for investigation; any link to a known event (e.g. Hurricane Sandy, 29-30 Oct 2012) "
             "is a temporal association - see reports/final_findings.md.")
    return Answer("anomalies", text, a.head(15), None, e)


def answer_trend(ctx: Context, e: Entities) -> Answer:
    k = analysis.kpis(ctx.df)
    mon = analysis.monthly_summary(ctx.df)
    best = mon.loc[mon["total"].idxmax()]
    text = (f"Total rentals grew from **{k['total_2011']:,}** in 2011 to **{k['total_2012']:,}** in 2012 "
            f"(**{k['yoy_growth_pct']:+.1f}%**). The busiest month was {best.month:%B %Y} ({best.total:,.0f} rentals). "
            "Growth matters for estimation: the model uses the year as an input and defaults to the 2012 level.")
    chart = {"kind": "line", "data": mon, "x": "month", "y": ["total"], "title": "Monthly rentals"}
    return Answer("trend", text, None, chart, e)


def answer_hourly(ctx: Context, e: Entities) -> Answer:
    return answer_compare_day_types(ctx, e) if not e.day_types else answer_peak(ctx, e)


def answer_overview(ctx: Context, e: Entities) -> Answer:
    k = analysis.kpis(ctx.df)
    text = (f"Across {k['n_days']} days ({k['n_hours']:,} hourly records, 2011-2012) there were **{k['total_rentals']:,}** rentals: "
            f"on average **{k['mean_daily']:.0f} per day** and **{k['mean_hourly']:.0f} per hour** (median {k['median_hourly']:.0f}). "
            f"The busiest single hour had {k['max_hourly']} rentals ({k['max_hourly_when']}). Registered users account for "
            f"{k['registered_share_pct']:.0f}% of rentals.")
    return Answer("overview", text, None, None, e)


def _scenario_base(ctx: Context, e: Entities) -> tuple[dict, list[str]]:
    df = ctx.df
    notes = []
    if e.month:
        season = analysis.typical_season_for_month(df, e.month)
        season_txt = f"month {e.month} (season code {season})"
    elif e.seasons:
        season = analysis.season_code(e.seasons[0], ctx.season_scheme)
        season_txt = f"{e.seasons[0]} (season code {season}, {analysis.SEASON_DATE_RANGES[season]})"
    else:
        season = analysis.season_code("summer", ctx.season_scheme)
        season_txt = f"summer (season code {season})"
        notes.append("No season given - assumed **summer** (current season interpretation).")
    wd = e.day_types[0] if e.day_types else 1
    if not e.day_types:
        notes.append("No day type given - assumed a **working day**.")
    if e.holiday:
        wd = 0
    weather = e.weathers[0] if e.weathers else 1
    if not e.weathers:
        notes.append("No weather given - assumed **clear** weather.")
    defaults = analysis.scenario_defaults(df, season, weather)
    temp = e.temp_c if e.temp_c is not None else defaults["temp_c"]
    hum = e.hum_pct if e.hum_pct is not None else defaults["hum_pct"]
    wind = e.wind_kmh if e.wind_kmh is not None else defaults["windspeed_kmh"]
    if e.temp_c is None:
        notes.append(f"Temperature not given - used the median observed for this season/weather ({temp:.1f} °C).")
    if e.hum_pct is None or e.wind_kmh is None:
        notes.append(f"Humidity/wind not given - used seasonal medians ({hum:.0f}% / {wind:.0f} km/h).")
    yr = (e.year - 2011) if e.year else 1
    if not e.year:
        notes.append("Demand level of **2012** assumed (most recent year in the data).")
    weekday = e.weekday
    if weekday is not None and wd == 1 and weekday in (0, 6):
        wd = 0
    params = dict(season=season, workingday=wd, weathersit=weather, temp_c=temp, hum_pct=hum,
                  windspeed_kmh=wind, yr=yr, mnth=e.month, weekday=weekday, holiday=int(e.holiday),
                  buffer_pct=ctx.buffer_pct)
    notes.insert(0, f"Season: {season_txt}.")
    return params, notes


def answer_predict(ctx: Context, e: Entities) -> Answer:
    from src.predict import day_profile, predict_scenario
    if ctx.bundle is None:
        return Answer("predict", "The trained model is not available. Run `python -m src.train_model` first.")
    base, notes = _scenario_base(ctx, e)
    if ctx.season_scheme == "official" and (e.seasons or not e.month):
        notes.append(_season_note("official"))
    if base["weathersit"] == 4 or 4 in e.weathers:
        notes.append("Heavy rain/snow appears in only 3 training hours; that estimate is highly uncertain.")

    # Which dimension varies? (weather, day type, hour or season - first one with >1 values)
    variants: list[tuple[str, dict]] = []
    if len(e.weathers) > 1:
        variants = [(WEATHER_LABELS[w], {"weathersit": w, **_weather_defaults(ctx, base, w, e)}) for w in e.weathers]
    elif len(e.day_types) > 1:
        variants = [("working day" if d else "non-working day", {"workingday": d}) for d in e.day_types]
    elif len(e.seasons) > 1:
        variants = [(s, {"season": analysis.season_code(s, ctx.season_scheme),
                         **analysis.scenario_defaults(ctx.df, analysis.season_code(s, ctx.season_scheme), base["weathersit"])})
                    for s in e.seasons]
        for _, v in variants:
            if e.temp_c is not None:
                v["temp_c"] = e.temp_c
            v["hum_pct"], v["windspeed_kmh"] = v.pop("hum_pct"), v.pop("windspeed_kmh")
    elif len(e.hours) > 1:
        variants = [(_hr(h), {"hr": h}) for h in e.hours]

    buffer_note = (f"Planning target adds a **{ctx.buffer_pct:.0f}% safety buffer** - an operational assumption you set in "
                   "the sidebar, not a model output." if ctx.buffer_pct else "")
    common = ("These are estimates of **system-wide rentals in that hour**, not the number of bikes needed at a station; "
              "fleet sizing also depends on trip durations, station capacity and rebalancing, which this dataset does not contain.")

    if variants:
        rows = []
        for label, over in variants:
            p = {**base, **over}
            if "hr" in p or e.hours:
                hr = p.pop("hr", e.hours[0] if e.hours else None)
            else:
                hr = None
            if hr is None:
                prof = day_profile(ctx.bundle, **p)
                rows.append({"scenario": label, "expected rentals (day total)": round(prof["estimate"].sum()),
                             "peak hour": _hr(int(prof.loc[prof["estimate"].idxmax(), "hr"])),
                             "peak-hour estimate": round(prof["estimate"].max())})
            else:
                r = predict_scenario(ctx.bundle, hr=hr, **p)
                rows.append({"scenario": label, "hour": _hr(hr), "expected rentals": r["estimate_rounded"],
                             "indicative 80% range": f"{r['range80'][0]:.0f}-{r['range80'][1]:.0f}",
                             **({"planning target": r["planning_target"]} if r["planning_target"] else {})})
        tab = pd.DataFrame(rows)
        key = "expected rentals" if "expected rentals" in tab else "expected rentals (day total)"
        hi_row, lo_row = tab.loc[tab[key].idxmax()], tab.loc[tab[key].idxmin()]
        diff = 100 * (lo_row[key] / hi_row[key] - 1) if hi_row[key] else 0
        scope = f"at {_hr(e.hours[0])}" if "hour" in tab and len(e.hours) <= 1 else ("for a full day" if key.endswith("total)") else "")
        text = (f"Model estimates {scope}: " + "; ".join(f"**{r['scenario']}** -> {r[key]:,}" for _, r in tab.iterrows())
                + f". **{lo_row['scenario']}** is {abs(diff):.0f}% lower than **{hi_row['scenario']}**, all else equal.")
        text += "\n\n**Assumptions:**\n" + "\n".join(f"- {n}" for n in notes)
        text += f"\n\n{common} {buffer_note}"
        return Answer("predict", text, tab, None, e)

    if not e.hours:
        prof = day_profile(ctx.bundle, **base)
        peak = prof.loc[prof["estimate"].idxmax()]
        text = (f"No hour was specified, so here is the **full-day profile**: expected **{prof['estimate'].sum():,.0f} rentals** "
                f"in total, peaking at {_hr(int(peak.hr))} (~{peak.estimate:.0f}/hour). Ask e.g. '... at 8 AM' for a single hour."
                "\n\n**Assumptions:**\n" + "\n".join(f"- {n}" for n in notes) + f"\n\n{common}")
        chart = {"kind": "bar", "data": prof.round(0), "x": "hr", "y": ["estimate"], "title": "Predicted rentals by hour"}
        return Answer("predict", text, None, chart, e)

    r = predict_scenario(ctx.bundle, hr=e.hours[0], **base)
    text = (f"Expected demand: **{r['estimate_rounded']} rentals** in the hour {r['inputs']['hour']} "
            f"(indicative 80% range {r['range80'][0]:.0f}-{r['range80'][1]:.0f}; the simple hour x working-day baseline "
            f"would say {r['baseline_estimate']:.0f}).")
    if r["planning_target"]:
        text += f" Planning target with {ctx.buffer_pct:.0f}% buffer: **{r['planning_target']}**."
    text += "\n\n**Assumptions:**\n" + "\n".join(f"- {n}" for n in notes) + f"\n\n{common} {buffer_note}"
    tab = pd.DataFrame([{"input": k, "value": str(v)} for k, v in r["inputs"].items()])
    return Answer("predict", text, tab, None, e)


def _weather_defaults(ctx: Context, base: dict, w: int, e: Entities) -> dict:
    """Keep temperature fixed across weather variants unless the season/weather default is all we have."""
    d = analysis.scenario_defaults(ctx.df, base["season"], w)
    out = {"hum_pct": e.hum_pct if e.hum_pct is not None else d["hum_pct"]}
    return out


HANDLERS = {
    "peak_hours": answer_peak, "compare_day_types": answer_compare_day_types, "weather_effect": answer_weather,
    "season_average": answer_season, "monitor_hours": answer_monitor, "hypothesis": answer_hypothesis,
    "model_performance": answer_model_perf, "anomalies": answer_anomalies, "trend": answer_trend,
    "hourly_pattern": answer_hourly, "overview": answer_overview, "predict": answer_predict,
}


def answer(question: str, ctx: Context) -> Answer:
    if not question or not question.strip():
        return help_answer(False, "unknown")
    e = extract_entities(question)
    intent = detect_intent(question, e)
    if intent == "unknown" and ctx.use_llm:
        intent = llm_intent(question) or "unknown"
    if intent == "help":
        return help_answer(True)
    if intent not in HANDLERS:
        return help_answer(False, "unknown")
    return HANDLERS[intent](ctx, e)

