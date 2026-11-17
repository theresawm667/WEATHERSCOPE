"""Feature engineering: turning raw daily rows into a forecasting table.

THE KEY IDEA
------------
One row of the output = "standing at the END of day D, in city C".
  * The INPUT columns only use information known by the end of day D.
  * The TARGET is the maximum temperature on day D+1 (tomorrow).
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_tmax_next_day"
CATEGORICAL_FEATURES = ["city"]
NUMERIC_FEATURES = [
    "tmax_today",
    "tmax_lag1",     # yesterday's max
    "tmax_lag2",     # two days ago
    "tmax_lag3",     # three days ago
    "tmax_mean7",    # average max over the last 7 days (including today)
    "tmin_today",
    "rain_today",
    "rain_sum7",     # total rain over the last 7 days (including today)
    "wind_today",
    "doy_sin",       # seasonality, part 1
    "doy_cos",       # seasonality, part 2
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Used by the app to explain every input and why it is legitimate to use.
FEATURE_GUIDE = [
    ("city", "Which city the forecast is for", "Known in advance"),
    ("tmax_today", "Today's maximum temperature (°C)", "Observed by the end of today"),
    ("tmax_lag1", "Yesterday's maximum temperature", "Past observation"),
    ("tmax_lag2", "Maximum temperature 2 days ago", "Past observation"),
    ("tmax_lag3", "Maximum temperature 3 days ago", "Past observation"),
    ("tmax_mean7", "Mean of the last 7 daily maxima", "Computed from past observations"),
    ("tmin_today", "Today's minimum temperature (°C)", "Observed by the end of today"),
    ("rain_today", "Today's rainfall (mm)", "Observed by the end of today"),
    ("rain_sum7", "Rainfall over the last 7 days (mm)", "Computed from past observations"),
    ("wind_today", "Today's maximum wind speed (km/h)", "Observed by the end of today"),
    ("doy_sin / doy_cos", "Time of year encoded as a circle", "Calendar: known in advance"),
]

# Plausible limits used to validate manual input (degrees C, mm, km/h).
TEMP_LIMITS = (0.0, 55.0)
MAX_DAILY_RAIN = 1000.0
MAX_WIND = 200.0


def day_of_year_encoding(dates: pd.Series | pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Encode the day of the year as a point on a circle (sin, cos).

    Why: a plain number 1..366 says December 31 and January 1 are far apart,
    but weather-wise they are neighbours. On a circle they are neighbours too.
    """
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    angle = 2 * np.pi * doy / 365.25
    return np.sin(angle), np.cos(angle)


def build_features(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Build the forecasting table from the cleaned daily data.

    Parameters
    ----------
    df : output of data.load_weather_data()
    require_target : if True (training), drop rows whose "tomorrow" is not in
        the data. If False (the app), keep them so the last day of the dataset
        can still be used as the starting point of a forecast.

    Returns one row per (city, day) with FEATURE_COLUMNS, `date` (day D),
    `target_date` (day D+1) and TARGET_COLUMN (NaN when tomorrow is unknown).
    """
    # Sorting matters: shift() works on row order, so rows must be in date order.
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    # groupby("city") makes every shift/rolling happen INSIDE one city. Without
    # it, Delhi's last day would borrow Hyderabad's first day as "tomorrow".
    by_city = df.groupby("city", sort=False)

    feats = pd.DataFrame({"city": df["city"], "date": df["date"]})
    tmax = "temperature_2m_max"

    feats["tmax_today"] = df[tmax]
    feats["tmax_lag1"] = by_city[tmax].shift(1)
    feats["tmax_lag2"] = by_city[tmax].shift(2)
    feats["tmax_lag3"] = by_city[tmax].shift(3)
    feats["tmax_mean7"] = by_city[tmax].transform(lambda s: s.rolling(7).mean())
    feats["tmin_today"] = df["temperature_2m_min"]
    feats["rain_today"] = df["rain_sum"]
    feats["rain_sum7"] = by_city["rain_sum"].transform(lambda s: s.rolling(7).sum())
    feats["wind_today"] = df["wind_speed_10m_max"]

    feats["doy_sin"], feats["doy_cos"] = day_of_year_encoding(df["date"])

    # Safety check: the 6 previous rows must really be the 6 previous days.
    # (Protects us if the CSV ever has missing days.)
    six_days_back = by_city["date"].shift(6)
    has_full_history = (df["date"] - six_days_back) == pd.Timedelta(days=6)
    feats = feats[has_full_history].copy()

    # The target: tomorrow's maximum, taken from the NEXT row of the same city.
    next_date = by_city["date"].shift(-1)[has_full_history]
    next_tmax = by_city[tmax].shift(-1)[has_full_history]
    tomorrow_is_next_row = (next_date - feats["date"]) == pd.Timedelta(days=1)

    feats["target_date"] = feats["date"] + pd.Timedelta(days=1)
    feats[TARGET_COLUMN] = next_tmax.where(tomorrow_is_next_row)

    if require_target:
        feats = feats.dropna(subset=[TARGET_COLUMN])

    return feats.reset_index(drop=True)


def make_feature_row(
    city: str,
    today: date | datetime | pd.Timestamp,
    tmax_last7: list[float],
    tmin_today: float,
    rain_today: float,
    rain_sum7: float,
    wind_today: float,
) -> pd.DataFrame:
    """Build ONE model-input row from values typed in by a person.

    tmax_last7 lists the maximum temperature for the last 7 days, oldest
    first, with today's value last.
    """
    if len(tmax_last7) != 7:
        raise ValueError("tmax_last7 must contain exactly 7 values (oldest first).")

    doy_sin, doy_cos = day_of_year_encoding(pd.DatetimeIndex([pd.Timestamp(today)]))
    return pd.DataFrame(
        {
            "city": [city],
            "tmax_today": [tmax_last7[6]],
            "tmax_lag1": [tmax_last7[5]],
            "tmax_lag2": [tmax_last7[4]],
            "tmax_lag3": [tmax_last7[3]],
            "tmax_mean7": [float(np.mean(tmax_last7))],
            "tmin_today": [tmin_today],
            "rain_today": [rain_today],
            "rain_sum7": [rain_sum7],
            "wind_today": [wind_today],
            "doy_sin": [doy_sin[0]],
            "doy_cos": [doy_cos[0]],
        }
    )[FEATURE_COLUMNS]


def validate_manual_inputs(
    tmax_last7: list[float],
    tmin_today: float,
    rain_today: float,
    rain_sum7: float,
    wind_today: float,
) -> list[str]:
    """Return a list of problems with manually entered values (empty = valid)."""
    problems: list[str] = []
    low, high = TEMP_LIMITS

    for label, value in zip(
        ["6 days ago", "5 days ago", "4 days ago", "3 days ago", "2 days ago", "yesterday", "today"],
        tmax_last7,
    ):
        if not (low <= value <= high):
            problems.append(f"Maximum temperature for {label} ({value} °C) is outside {low}–{high} °C.")

    if not (low <= tmin_today <= high):
        problems.append(f"Today's minimum ({tmin_today} °C) is outside {low}–{high} °C.")
    if tmin_today > tmax_last7[6]:
        problems.append("Today's minimum temperature cannot be higher than today's maximum.")
    if rain_today < 0 or rain_today > MAX_DAILY_RAIN:
        problems.append(f"Today's rainfall must be between 0 and {MAX_DAILY_RAIN:.0f} mm.")
    if rain_sum7 < rain_today:
        problems.append("The 7-day rainfall total cannot be smaller than today's rainfall.")
    if wind_today < 0 or wind_today > MAX_WIND:
        problems.append(f"Wind speed must be between 0 and {MAX_WIND:.0f} km/h.")

    return problems
