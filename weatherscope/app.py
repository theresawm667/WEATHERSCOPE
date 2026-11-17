"""WeatherScope: Streamlit interface.

Start it from the project folder with:
    streamlit run app.py

This file only arranges pages and widgets. The real work happens in:
data.py (loading), features.py (inputs), models.py, evaluate.py (metrics),
analysis.py (statistics) and plots.py (charts).
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import analysis
import plots
from data import DataError, load_weather_data, validate_weather_data
from evaluate import VALIDATION_END, climatology_predictions, split_by_target_date
from features import (
    FEATURE_COLUMNS, FEATURE_GUIDE, TARGET_COLUMN, TEMP_LIMITS,
    build_features, make_feature_row, validate_manual_inputs,
)
from models import MODEL_LABELS, ModelError, load_model
from paths import METRICS_PATH, PREDICTIONS_PATH

st.set_page_config(page_title="WeatherScope", page_icon="🌦️", layout="wide")

PAGES = ["Overview", "Explore", "City Compare", "Model Lab", "Predict"]
NO_RESULTS_HELP = (
    "No training results found yet. Open a terminal in the project folder and run:\n\n"
    "`python train.py`\n\nThen refresh this page."
)


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading weather data...")
def get_data() -> pd.DataFrame:
    return load_weather_data()


@st.cache_data(show_spinner=False)
def get_features() -> pd.DataFrame:
    # require_target=False keeps the last day of the data as a forecast start.
    return build_features(get_data(), require_target=False)


@st.cache_data(show_spinner=False)
def get_metrics(mtime_key: float) -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def get_test_predictions(mtime_key: float) -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["date", "target_date"])


@st.cache_resource(show_spinner="Loading model...")
def get_model(key: str):
    return load_model(key)


def header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)


# Page 1: Overview

def page_overview(df: pd.DataFrame, metrics: dict | None) -> None:
    st.markdown(
        """
        <div style="padding:1.4rem 1.6rem;border-radius:14px;
        background:linear-gradient(120deg,#0E7C86,#14A3A8);color:white;">
        <h1 style="margin:0;color:white;">🌦️ WeatherScope</h1>
        <p style="margin:.3rem 0 0 0;font-size:1.05rem;">
        Explore 25 years of daily weather for 10 Indian cities, and test how well a
        machine-learning model can forecast <b>tomorrow's maximum temperature</b>.</p></div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily records", f"{len(df):,}")
    c2.metric("Cities", df["city"].nunique())
    c3.metric("From", df["date"].min().strftime("%d %b %Y"))
    c4.metric("To", df["date"].max().strftime("%d %b %Y"))

    left, right = st.columns(2)
    with left:
        st.markdown("### What can I do here?")
        st.markdown(
            "- **Explore**: filter by city and date range, see trends, distributions and seasonality.\n"
            "- **City Compare**: rank cities by heat, rainfall and monsoon share.\n"
            "- **Model Lab**: see how three models compare against simple baselines, and why evaluation design matters.\n"
            "- **Predict**: forecast tomorrow's maximum temperature from the last few days."
        )
    with right:
        st.markdown("### An honest note on what the model does")
        st.info(
            "The model predicts **next-day maximum temperature** from recent observations of the same city. "
            "It learns from past records only; it does **not** use weather forecasts, satellite data or "
            "atmospheric physics, so it is a learning exercise, not an official forecast. "
            "The Model Lab shows how it compares with trivial baselines."
        )

    st.markdown("### Inputs the model uses (and why they are fair to use)")
    guide = pd.DataFrame(FEATURE_GUIDE, columns=["Input", "Meaning", "Available when forecasting?"])
    st.dataframe(guide, hide_index=True)

    warnings = validate_weather_data(df)
    with st.expander("Data quality checks"):
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("No missing values, duplicates or gaps found in the daily sequence.")


# Page 2: Explore

def page_explore(df: pd.DataFrame) -> None:
    header("Explore", "Filter the data, then look at trends, distributions and seasonality.")
    cities = sorted(df["city"].unique())
    default = [c for c in ("Delhi", "Mumbai") if c in cities] or cities[:2]

    c1, c2, c3 = st.columns([2, 2, 1.5])
    chosen = c1.multiselect("Cities", cities, default=default)
    label = c2.selectbox("Variable", list(analysis.VARIABLES))
    freq = c3.radio("Aggregation", ["Daily", "Monthly", "Yearly"], index=1, horizontal=True)

    lo, hi = df["date"].min().date(), df["date"].max().date()
    span = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)
    if not isinstance(span, tuple) or len(span) != 2:
        st.info("Pick both a start and an end date.")
        return
    if not chosen:
        st.info("Select at least one city.")
        return

    column = analysis.VARIABLES[label]
    mask = df["city"].isin(chosen) & (df["date"] >= pd.Timestamp(span[0])) & (df["date"] <= pd.Timestamp(span[1]))
    sub = df[mask]
    if sub.empty:
        st.warning("No data for this selection.")
        return

    tab_trend, tab_dist, tab_season, tab_stats = st.tabs(["Trend", "Distribution", "Seasonality", "Statistics"])

    with tab_trend:
        series = analysis.aggregate_series(sub, column, freq)
        agg_word = "total" if column == "rain_sum" else "mean"
        title = label if freq == "Daily" else f"{freq} {agg_word}: {label}"
        st.plotly_chart(plots.line_chart(series, "date", column, "city", title, label))
        if freq != "Daily":
            st.caption("The first and last periods may be partial if your date range does not start/end on a month or year boundary.")
        if column in ("temperature_2m_max", "temperature_2m_min"):
            trend = analysis.yearly_trend_per_decade(sub, column)
            if not trend.empty:
                st.markdown("**Straight-line trend of yearly averages (°C per decade)**")
                st.dataframe(trend, hide_index=True)
                st.caption("Descriptive only: it does not explain why (urban growth, climate change, measurement differences...).")

    with tab_dist:
        if column == "rain_sum":
            wet = sub[sub["rain_sum"] > 0]
            st.plotly_chart(plots.histogram(wet, column, "city", "Rainfall on days with any rain",
                                            "Rain (mm/day), log count axis", log_y=True))
            dry_pct = 100 * (sub["rain_sum"] == 0).groupby(sub["city"]).mean()
            st.markdown("**Share of completely dry days:** " + ", ".join(f"{c} {v:.0f}%" for c, v in dry_pct.items()))
            st.caption("Rainfall is heavily skewed: most days are dry and a few days carry very large amounts, which is why a log axis is used.")
        else:
            st.plotly_chart(plots.histogram(sub, column, "city", f"Distribution of daily values: {label}", label))

    with tab_season:
        profile = analysis.monthly_profile(sub, column)
        y_label = "Typical monthly total (mm)" if column == "rain_sum" else f"Average {label}"
        st.plotly_chart(plots.month_lines(profile, column, f"By calendar month: {label}", y_label))

    with tab_stats:
        st.dataframe(analysis.describe_by_city(sub, column), hide_index=True)
        st.caption("5th/95th pct = the value below which 5%/95% of days fall. A large gap between mean and median signals a skewed variable.")


# Page 3: City Compare

def page_compare(df: pd.DataFrame) -> None:
    header("City Compare", "Side-by-side climate statistics over the whole 2000-2024 record.")
    cities = sorted(df["city"].unique())
    c1, c2 = st.columns([3, 1.5])
    chosen = c1.multiselect("Cities to compare", cities, default=cities[:4])
    threshold = c2.slider("'Hot day' threshold (°C)", 35.0, 45.0, 40.0, 0.5)
    if len(chosen) < 2:
        st.info("Select at least two cities to compare.")
        return
    sub = df[df["city"].isin(chosen)]

    st.markdown("#### Headline statistics")
    st.dataframe(analysis.city_summary(sub, threshold), hide_index=True)
    st.caption(f"A 'rainy day' means at least {analysis.RAINY_DAY_MM} mm of rain. Monsoon = June-September.")

    st.markdown("#### Insights")
    for note in analysis.key_insights(df, chosen):
        st.markdown(f"- {note}")

    tab_heat, tab_season = st.tabs(["Month × city heatmap", "Seasons"])
    with tab_heat:
        which = st.radio("Show", ["Max temperature", "Min temperature", "Rainfall (monthly total)"], horizontal=True)
        col, unit, scale = {
            "Max temperature": ("temperature_2m_max", "°C", "YlOrRd"),
            "Min temperature": ("temperature_2m_min", "°C", "YlGnBu"),
            "Rainfall (monthly total)": ("rain_sum", "mm", "Blues"),
        }[which]
        matrix = analysis.city_month_matrix(sub, col)
        st.plotly_chart(plots.heatmap(matrix, f"{which} by month", unit, scale))
    with tab_season:
        seasons = analysis.season_summary(sub)
        st.plotly_chart(plots.season_bars(seasons, "avg_max", "Average maximum temperature by season", "°C"))
        st.plotly_chart(plots.season_bars(seasons, "avg_daily_rain", "Average daily rainfall by season", "mm/day"))
        st.dataframe(seasons.rename(columns={"avg_max": "avg max (°C)", "avg_min": "avg min (°C)",
                                             "avg_daily_rain": "avg rain (mm/day)"}), hide_index=True)


# Page 4: Model Lab

def _skill_sentence(metrics: dict) -> str:
    best = metrics["best_model"]
    m = metrics["models"][best]["test"]["mae"]
    persistence = metrics["baselines"]["persistence"]["test"]["mae"]
    gain = 100 * (1 - m / persistence)
    name = metrics["models"][best]["label"]
    if gain > 0:
        return (f"On the held-out test period, **{name}** has an average error of **{m:.2f} °C**, "
                f"which is **{gain:.0f}% lower** than simply assuming tomorrow equals today ({persistence:.2f} °C).")
    return (f"On the held-out test period, **{name}** has an average error of {m:.2f} °C, which does "
            f"**not** beat simply assuming tomorrow equals today ({persistence:.2f} °C).")


def page_model_lab(metrics: dict | None) -> None:
    header("Model Lab", "How good is the model, measured honestly?")
    if metrics is None:
        st.warning(NO_RESULTS_HELP)
        return
    meta = metrics["meta"]
    st.info(f"Trained on **{meta['train_period']}**, compared on **{meta['validation_period']}**, "
            f"final scores on **{meta['test_period']}** (data the final models never saw). "
            f"Results created {meta['trained_at_utc'][:10]} with scikit-learn {meta['sklearn_version']}.")
    st.markdown(_skill_sentence(metrics))

    tab_cmp, tab_detail, tab_imp, tab_leak = st.tabs(
        ["Model comparison", "Accuracy details", "Feature importance", "Why evaluation matters"]
    )

    with tab_cmp:
        rows = []
        for key, b in metrics["baselines"].items():
            rows.append({"name": b["label"], "kind": "Baseline", "key": key, **{f"val_{k}": v for k, v in b["validation"].items()},
                         **{f"test_{k}": v for k, v in b["test"].items()}})
        for key, m in metrics["models"].items():
            rows.append({"name": m["label"], "kind": "Model", "key": key, **{f"val_{k}": v for k, v in m["validation"].items()},
                         **{f"test_{k}": v for k, v in m["test"].items()}})
        table = pd.DataFrame(rows)
        shown = table[["name", "kind", "val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2"]].rename(columns={
            "name": "Method", "kind": "Type", "val_mae": "Val MAE", "val_rmse": "Val RMSE", "val_r2": "Val R²",
            "test_mae": "Test MAE", "test_rmse": "Test RMSE", "test_r2": "Test R²"})
        st.dataframe(shown.round(3), hide_index=True)
        st.caption("MAE and RMSE are in °C (lower is better); R² is 0-1 (higher is better). "
                   "Validation scores were used to pick the best model; test scores are the final, untouched check.")
        metric = st.radio("Chart metric", ["test_mae", "test_rmse"], horizontal=True,
                          format_func=lambda m: "Test MAE" if m == "test_mae" else "Test RMSE")
        st.plotly_chart(plots.model_comparison(table, metric, "Test-period error (lower is better)"))
        st.success(f"Best model on validation RMSE: **{metrics['models'][metrics['best_model']]['label']}**")

    with tab_detail:
        options = list(metrics["models"])
        key = st.selectbox("Model", options, index=options.index(metrics["best_model"]),
                           format_func=lambda k: MODEL_LABELS.get(k, k))
        info = metrics["models"][key]
        q = info["residual_quantiles"]
        st.write(f"Typical error range: 90% of test days were between **{q['q05']:+.1f} °C** and **{q['q95']:+.1f} °C** "
                 "(actual − predicted).")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(plots.bar(pd.DataFrame(info["by_city"]), "city", "mae", "Test MAE by city", "MAE (°C)"))
        with right:
            by_month = pd.DataFrame(info["by_month"])
            by_month["month"] = by_month["month"].map(lambda m: analysis.MONTH_NAMES[int(m) - 1])
            st.plotly_chart(plots.bar(by_month, "month", "mae", "Test MAE by month of the forecast day", "MAE (°C)"))
        preds = get_test_predictions(_mtime(PREDICTIONS_PATH))
        if preds is not None and key in preds.columns:
            sample = preds.sample(min(4000, len(preds)), random_state=1)
            a, b = st.columns(2)
            a.plotly_chart(plots.actual_vs_predicted(sample["actual"], sample[key], "Actual vs predicted (test period)"))
            b.plotly_chart(plots.residual_histogram(preds["actual"] - preds[key], "Distribution of errors"))

    with tab_imp:
        imp = pd.DataFrame(metrics["feature_importance"])
        best_label = metrics["models"][metrics["best_model"]]["label"]
        st.plotly_chart(plots.bar(imp, "feature", "mae_increase",
                                  f"Permutation importance: {best_label}", "Increase in MAE (°C) when shuffled",
                                  horizontal=True))
        st.markdown(
            "**How to read this:** each input is randomly shuffled in turn. If the error rises a lot, the model was relying on it. "
            "Importance shows what the model *uses*, not what *causes* tomorrow's weather. Inputs that are correlated "
            "(e.g. today's max and the 7-day mean) can share importance between them."
        )

    with tab_leak:
        st.markdown(
            "The project this one grew from estimated **today's** maximum from **today's** other measurements "
            "(including apparent temperature, which is calculated partly from the temperature itself) using a random "
            "train/test split. The table below re-creates that setup and then evaluates it more strictly. "
            "This is a **different task** from forecasting tomorrow, so don't compare its numbers with the table on the first tab."
        )
        demo = pd.DataFrame(metrics["leakage_demo"]).rename(
            columns={"id": "ID", "setup": "Evaluation setup", "mae": "MAE (°C)", "rmse": "RMSE (°C)", "r2": "R²"})
        st.dataframe(demo.round(3), hide_index=True)
        st.markdown(
            "- **Row A** looks excellent, but the random split mixes neighbouring days into training and test, "
            "and same-day apparent temperature almost gives away the answer.\n"
            "- Moving to a **chronological split** (B) is more honest about future data.\n"
            "- Removing inputs that would not exist before the day (C, D) shows the real difficulty.\n"
            "- A **baseline** with no machine learning shows how much the model actually adds."
        )

    st.markdown("#### Limitations")
    st.markdown(
        "- The model sees only past observations from one city at a time; it cannot know about an approaching "
        "storm system, so sudden changes are its biggest errors.\n"
        "- Forecast skill is judged on a 2022-2024 test window; other periods may differ.\n"
        "- Rain is treated as a plain number; the model does not model weather regimes explicitly.\n"
        "- Accuracy is lower for the months when weather changes quickly (see the by-month chart)."
    )


# Page 5: Predict

def _error_band(metrics: dict | None, key: str, forecast: float) -> str | None:
    if metrics is None or key not in metrics["models"]:
        return None
    q = metrics["models"][key]["residual_quantiles"]
    return f"{forecast + q['q05']:.1f} to {forecast + q['q95']:.1f} °C"


def _show_forecast_cards(forecast: float, persistence: float, climatology: float, actual: float | None,
                         band: str | None) -> None:
    cols = st.columns(4 if actual is not None else 3)
    cols[0].metric("Model forecast", f"{forecast:.1f} °C")
    cols[1].metric("'Same as today'", f"{persistence:.1f} °C")
    cols[2].metric("City + month average", f"{climatology:.1f} °C")
    if actual is not None:
        cols[3].metric("Actual (dataset)", f"{actual:.1f} °C")
        errors = {"Model": abs(forecast - actual), "'Same as today'": abs(persistence - actual),
                  "City + month avg": abs(climatology - actual)}
        st.write("Absolute error: " + " | ".join(f"{k} **{v:.2f} °C**" for k, v in errors.items()))
        st.caption("A single day is just one example; it cannot tell you which method is better. "
                   "Use the Model Lab for the full-period comparison.")
    if band:
        st.caption(f"Typical uncertainty: in the test period, 90% of this model's errors would put the "
                   f"actual temperature between **{band}**.")


def page_predict(df: pd.DataFrame, metrics: dict | None) -> None:
    header("Predict", "Forecast the NEXT day's maximum temperature from the last few days.")
    if metrics is None:
        st.warning(NO_RESULTS_HELP)
        return

    available = [k for k in metrics["models"] if k in MODEL_LABELS]
    model_key = st.selectbox("Model", available, index=available.index(metrics["best_model"]),
                             format_func=lambda k: MODEL_LABELS[k])
    try:
        model = get_model(model_key)
    except ModelError as exc:
        st.error(str(exc))
        return

    feats = get_features()
    reference = feats[(feats["target_date"] <= VALIDATION_END) & feats[TARGET_COLUMN].notna()]
    tab_hist, tab_manual = st.tabs(["Historical demonstration", "Enter values manually"])

    with tab_hist:
        st.write("Pick a city and a day from the dataset. The model sees **only** that day and the days before it, "
                 "then forecasts the day after.")
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", sorted(df["city"].unique()), key="hist_city")
        city_feats = feats[feats["city"] == city]
        lo, hi = city_feats["date"].min().date(), city_feats["date"].max().date()
        day = c2.date_input("Forecast from (the 'today' day)", value=date(2023, 5, 15),
                            min_value=lo, max_value=hi, key="hist_date")
        row = city_feats[city_feats["date"] == pd.Timestamp(day)]
        if row.empty:
            st.warning("No complete history for that day. Try another date.")
        else:
            row = row.iloc[[0]]
            target_day = row["target_date"].iloc[0]
            actual = row[TARGET_COLUMN].iloc[0]
            actual = None if pd.isna(actual) else float(actual)
            forecast = float(model.predict(row[FEATURE_COLUMNS])[0])
            persistence = float(row["tmax_today"].iloc[0])
            climatology = float(climatology_predictions(reference, row)[0])

            st.markdown(f"#### Forecast for {city} on {target_day.strftime('%d %B %Y')}")
            if target_day > VALIDATION_END:
                st.success("This day is in the held-out test period: the saved model never saw it during training.")
            else:
                st.warning("This day was part of the data the model was trained on, so the result flatters the model. "
                           "Choose a date from 2022 onwards for a fair demonstration.")
            if actual is None:
                st.info("The dataset ends here, so there is no actual value to compare with.")
            _show_forecast_cards(forecast, persistence, climatology, actual, _error_band(metrics, model_key, forecast))

            hist = df[(df["city"] == city) & (df["date"] <= pd.Timestamp(day)) &
                      (df["date"] > pd.Timestamp(day) - pd.Timedelta(days=14))]
            history = pd.DataFrame({"date": hist["date"], "tmax": hist["temperature_2m_max"]})
            st.plotly_chart(plots.forecast_context(history, target_day, forecast, actual))

    with tab_manual:
        st.write("Type in recent observations. Starting values are typical for the chosen city and month; "
                 "replace them with your own numbers.")
        c1, c2 = st.columns(2)
        city_m = c1.selectbox("City", sorted(df["city"].unique()), key="man_city")
        today = c2.date_input("Today's date", value=date(2024, 6, 15), key="man_date",
                              min_value=date(2000, 1, 1), max_value=date(2035, 12, 31))
        typical = df[(df["city"] == city_m) & (df["date"].dt.month == today.month)]
        t_max, t_min = float(typical["temperature_2m_max"].mean()), float(typical["temperature_2m_min"].mean())
        t_wind = float(typical["wind_speed_10m_max"].mean())
        suffix = f"{city_m}_{today.month}"   # changing city/month resets the defaults

        st.markdown("**Maximum temperature (°C) for the last 7 days**")
        labels = ["6 days ago", "5 days ago", "4 days ago", "3 days ago", "2 days ago", "Yesterday", "Today"]
        cols = st.columns(7)
        tmax_last7 = [
            cols[i].number_input(labels[i], value=round(t_max, 1), step=0.5, format="%.1f", key=f"tmax{i}_{suffix}")
            for i in range(7)
        ]
        a, b, c, d = st.columns(4)
        tmin_today = a.number_input("Today's minimum (°C)", value=round(t_min, 1), step=0.5, format="%.1f", key=f"tmin_{suffix}")
        rain_today = b.number_input("Today's rain (mm)", value=0.0, min_value=0.0, step=0.5, key=f"rain_{suffix}")
        rain7 = c.number_input("Rain, last 7 days (mm)", value=0.0, min_value=0.0, step=0.5, key=f"rain7_{suffix}")
        wind = d.number_input("Today's max wind (km/h)", value=round(t_wind, 1), min_value=0.0, step=1.0, key=f"wind_{suffix}")

        if st.button("Forecast tomorrow's maximum", type="primary"):
            problems = validate_manual_inputs(tmax_last7, tmin_today, rain_today, rain7, wind)
            if problems:
                st.error("Please fix these inputs first:\n\n" + "\n".join(f"- {p}" for p in problems))
            else:
                row = make_feature_row(city_m, today, tmax_last7, tmin_today, rain_today, rain7, wind)
                forecast = float(model.predict(row)[0])
                ref_row = pd.DataFrame({"city": [city_m],
                                        "target_date": [pd.Timestamp(today) + pd.Timedelta(days=1)]})
                climatology = float(climatology_predictions(reference, ref_row)[0])
                city_range = df.loc[df["city"] == city_m, "temperature_2m_max"].agg(["min", "max"])
                if min(tmax_last7) < city_range["min"] - 3 or max(tmax_last7) > city_range["max"] + 3:
                    st.warning(f"Some temperatures are outside the range ever recorded in {city_m} "
                               f"({city_range['min']:.0f}-{city_range['max']:.0f} °C). "
                               "The model has not seen such values, so treat the result with extra caution.")
                st.markdown(f"#### Forecast for {city_m} on {(pd.Timestamp(today) + pd.Timedelta(days=1)).strftime('%d %B %Y')}")
                _show_forecast_cards(forecast, float(tmax_last7[-1]), climatology, None,
                                     _error_band(metrics, model_key, forecast))


# Main: sidebar navigation + page dispatch

def main() -> None:
    try:
        df = get_data()
    except DataError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.markdown("## 🌦️ WeatherScope")
        page = st.radio("Go to", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Historical data, 10 Indian cities, 2000-2024. "
                   "A learning project, not an official forecast.")

    metrics = get_metrics(_mtime(METRICS_PATH))
    if page == "Overview":
        page_overview(df, metrics)
    elif page == "Explore":
        page_explore(df)
    elif page == "City Compare":
        page_compare(df)
    elif page == "Model Lab":
        page_model_lab(metrics)
    else:
        page_predict(df, metrics)


main()
