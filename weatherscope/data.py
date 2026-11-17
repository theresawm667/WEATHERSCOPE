"""Loading and checking the weather CSV.

Everything that touches the raw file lives here, so the rest of the project can
assume it receives a clean, sorted DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import DATA_PATH

REQUIRED_COLUMNS = [
    "city",
    "date",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
]

NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in ("city", "date")]


class DataError(Exception):
    """Raised with a human-readable message when the dataset cannot be used."""


def load_weather_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and return a clean DataFrame sorted by city and date.

    Raises DataError (with advice on how to fix it) instead of a cryptic
    pandas traceback when the file is missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Dataset not found at:\n{path}\n\n"
            "Make sure 'weather_data.csv' is inside the 'data' folder of the "
            "project, and that you opened the project folder in VS Code."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable / corrupted file
        raise DataError(f"Could not read {path.name}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            f"The dataset is missing required columns: {', '.join(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows with an unreadable date or missing core measurements cannot be used.
    core = ["city", "date", "temperature_2m_max", "temperature_2m_min", "rain_sum"]
    df = df.dropna(subset=core)
    df = df.drop_duplicates(subset=["city", "date"])

    if df.empty:
        raise DataError("The dataset contains no usable rows after cleaning.")

    return df.sort_values(["city", "date"]).reset_index(drop=True)


def validate_weather_data(df: pd.DataFrame) -> list[str]:
    """Return a list of warnings about suspicious values (empty = all good)."""
    warnings: list[str] = []

    n_missing = int(df[NUMERIC_COLUMNS].isna().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing numeric values found in the dataset.")

    if (df["rain_sum"] < 0).any():
        warnings.append("Negative rainfall values found.")

    if (df["temperature_2m_min"] > df["temperature_2m_max"]).any():
        warnings.append("Some days have a minimum temperature above the maximum.")

    # Consecutive-day check: our lag features assume one row per calendar day.
    for city, group in df.groupby("city"):
        gaps = group["date"].diff().dropna()
        n_gaps = int((gaps != pd.Timedelta(days=1)).sum())
        if n_gaps:
            warnings.append(f"{city}: {n_gaps} gap(s) in the daily sequence.")

    return warnings
