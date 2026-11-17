"""Descriptive statistics for the Explore and City Compare pages.

Pure pandas functions: they take a DataFrame and return a DataFrame/list, so
they can be tested without Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Indian meteorological seasons (month -> season).
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
RAINY_DAY_MM = 2.5  # a day counts as "rainy" at or above this much rain

VARIABLES = {
    "Maximum temperature (°C)": "temperature_2m_max",
    "Minimum temperature (°C)": "temperature_2m_min",
    "Rainfall (mm/day)": "rain_sum",
    "Max wind speed (km/h)": "wind_speed_10m_max",
}


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["season"] = out["month"].map(SEASON_OF_MONTH)
    return out


def aggregate_series(df: pd.DataFrame, column: str, freq: str) -> pd.DataFrame:
    """Aggregate a variable per city: freq is 'Daily', 'Monthly' or 'Yearly'.

    Rainfall is SUMMED (a monthly total is meaningful); everything else is
    averaged.
    """
    how = "sum" if column == "rain_sum" else "mean"
    if freq == "Daily":
        return df[["city", "date", column]].copy()
    pandas_freq = {"Monthly": "MS", "Yearly": "YS"}[freq]
    out = (
        df.groupby(["city", pd.Grouper(key="date", freq=pandas_freq)])[column]
        .agg(how)
        .reset_index()
    )
    return out


def describe_by_city(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summary statistics per city: mean, spread, extremes and percentiles."""
    g = df.groupby("city")[column]
    table = pd.DataFrame(
        {
            "days": g.count(),
            "mean": g.mean(),
            "std dev": g.std(),
            "min": g.min(),
            "5th pct": g.quantile(0.05),
            "median": g.median(),
            "95th pct": g.quantile(0.95),
            "max": g.max(),
        }
    )
    return table.round(2).reset_index()


def monthly_profile(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Average of a variable for each calendar month, per city."""
    out = df.assign(month=df["date"].dt.month).groupby(["city", "month"])[column].mean()
    out = out.reset_index()
    if column == "rain_sum":
        # mean DAILY rain x days is awkward; show typical monthly TOTAL instead
        years = df["date"].dt.year.nunique()
        totals = df.assign(month=df["date"].dt.month).groupby(["city", "month"])[column].sum()
        out[column] = (totals / years).to_numpy()
    out["month_name"] = out["month"].map(lambda m: MONTH_NAMES[m - 1])
    return out


def yearly_trend_per_decade(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Straight-line trend of the yearly mean, expressed per 10 years.

    Purely descriptive: it does not say WHY a trend exists (urban growth,
    climate change, sensor changes in the underlying data...).
    """
    yearly = df.assign(year=df["date"].dt.year).groupby(["city", "year"])[column].mean().reset_index()
    rows = []
    for city, g in yearly.groupby("city"):
        if g["year"].nunique() < 3:
            continue
        slope = np.polyfit(g["year"], g[column], 1)[0]
        rows.append({"city": city, "change per decade": round(slope * 10, 2)})
    return pd.DataFrame(rows)


def city_summary(df: pd.DataFrame, hot_threshold: float = 40.0) -> pd.DataFrame:
    """One row per city with the headline climate statistics."""
    d = add_time_columns(df)
    n_years = d["year"].nunique()
    rows = []
    for city, g in d.groupby("city"):
        total_rain = g["rain_sum"].sum()
        monsoon_rain = g.loc[g["season"] == "Monsoon", "rain_sum"].sum()
        rows.append(
            {
                "city": city,
                "avg max (°C)": g["temperature_2m_max"].mean(),
                "avg min (°C)": g["temperature_2m_min"].mean(),
                "daily range (°C)": (g["temperature_2m_max"] - g["temperature_2m_min"]).mean(),
                f"days ≥ {hot_threshold:g}°C per year": (g["temperature_2m_max"] >= hot_threshold).sum() / n_years,
                "rain per year (mm)": total_rain / n_years,
                f"rainy days per year (≥ {RAINY_DAY_MM} mm)": (g["rain_sum"] >= RAINY_DAY_MM).sum() / n_years,
                "monsoon share of rain (%)": 100 * monsoon_rain / total_rain if total_rain else np.nan,
            }
        )
    return pd.DataFrame(rows).round(1)


def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per city and season: average max, average min and average daily rain."""
    d = add_time_columns(df)
    out = (
        d.groupby(["city", "season"])
        .agg(avg_max=("temperature_2m_max", "mean"),
             avg_min=("temperature_2m_min", "mean"),
             avg_daily_rain=("rain_sum", "mean"))
        .reset_index()
    )
    out["season"] = pd.Categorical(out["season"], SEASON_ORDER, ordered=True)
    return out.sort_values(["city", "season"]).round(2)


def city_month_matrix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """City x month table (used for the heatmap). Rain = typical monthly total."""
    prof = monthly_profile(df, column)
    matrix = prof.pivot(index="city", columns="month", values=column)
    matrix.columns = [MONTH_NAMES[m - 1] for m in matrix.columns]
    return matrix


def key_insights(df: pd.DataFrame, cities: list[str]) -> list[str]:
    """Plain-English facts computed from the data for the selected cities."""
    sub = df[df["city"].isin(cities)]
    if sub.empty:
        return []
    notes = []
    summary = city_summary(sub)
    hottest = summary.loc[summary["avg max (°C)"].idxmax()]
    coolest = summary.loc[summary["avg max (°C)"].idxmin()]
    wettest = summary.loc[summary["rain per year (mm)"].idxmax()]
    driest = summary.loc[summary["rain per year (mm)"].idxmin()]
    notes.append(f"Highest average daily maximum: **{hottest['city']}** ({hottest['avg max (°C)']} °C); "
                 f"lowest: **{coolest['city']}** ({coolest['avg max (°C)']} °C).")
    notes.append(f"Most rain per year: **{wettest['city']}** (~{wettest['rain per year (mm)']:.0f} mm); "
                 f"least: **{driest['city']}** (~{driest['rain per year (mm)']:.0f} mm).")

    month_means = sub.groupby(sub["date"].dt.month)["temperature_2m_max"].mean()
    notes.append(f"Across the selected cities, the hottest month on average is "
                 f"**{MONTH_NAMES[month_means.idxmax() - 1]}** and the coolest is "
                 f"**{MONTH_NAMES[month_means.idxmin() - 1]}**.")
    return notes
