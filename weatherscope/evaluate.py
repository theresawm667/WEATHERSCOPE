"""Measuring how good the models are, honestly.

Contains: the chronological split, the three metrics, two simple baselines,
error breakdowns, permutation feature importance, and a demonstration of how
target leakage inflates scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import FEATURE_COLUMNS, TARGET_COLUMN

# Chronological split, decided by the date being predicted (the target date).
TRAIN_END = pd.Timestamp("2019-12-31")        # train:      2000 - 2019
VALIDATION_END = pd.Timestamp("2021-12-31")   # validation: 2020 - 2021
                                              # test:       2022 - 2024


def split_by_target_date(feats: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks selecting the train / validation / test rows."""
    d = feats["target_date"]
    return {
        "train": d <= TRAIN_END,
        "validation": (d > TRAIN_END) & (d <= VALIDATION_END),
        "test": d > VALIDATION_END,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R², the three metrics used throughout the project."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def persistence_predictions(feats: pd.DataFrame) -> np.ndarray:
    """Baseline 1: 'tomorrow will be as hot as today'. Needs no training."""
    return feats["tmax_today"].to_numpy()


def climatology_predictions(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Baseline 2: the average max for that city and calendar month.

    Averages are computed ONLY from `reference` (past data), never from the
    rows being predicted.
    """
    ref_month = reference["target_date"].dt.month
    means = reference.groupby(["city", ref_month])[TARGET_COLUMN].mean()
    keys = pd.MultiIndex.from_arrays([target["city"], target["target_date"].dt.month])
    return means.reindex(keys).to_numpy()


def error_breakdown(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAE / RMSE per group. `frame` needs columns y_true, y_pred and `by`."""
    rows = []
    for key, g in frame.groupby(by):
        m = regression_metrics(g["y_true"], g["y_pred"])
        rows.append({by: key, "n": len(g), **m})
    return pd.DataFrame(rows)


def importance_table(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance: shuffle one column at a time and see how much
    the error (MAE) gets worse. A bigger increase = the model relies on it more.
    """
    result = permutation_importance(
        model, X[FEATURE_COLUMNS], y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mae_increase": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return table.sort_values("mae_increase", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage demonstration
# ---------------------------------------------------------------------------

_ORIGINAL_FEATURES = [
    "city", "temperature_2m_min", "apparent_temperature_max",
    "apparent_temperature_min", "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "year", "month", "day",
]
_SAME_DAY_TEMPERATURE = {
    "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min",
}


def leakage_demo(raw: pd.DataFrame) -> list[dict]:
    #Show how the SAME-DAY estimation task scores under different evaluation choices.
    df = raw.copy()
    df["year"], df["month"], df["day"] = df["date"].dt.year, df["date"].dt.month, df["date"].dt.day
    target = "temperature_2m_max"
    time_train = df["date"] < "2020-01-01"

    def fit_score(features, split):
        if split == "random":
            train, test = train_test_split(df, test_size=0.2, random_state=42)
        else:
            train, test = df[time_train], df[~time_train]
        prep = ColumnTransformer(
            [("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"])],
            remainder="passthrough",
        )
        forest = RandomForestRegressor(n_estimators=60, min_samples_leaf=3, n_jobs=-1, random_state=42)
        pipe = Pipeline([("prep", prep), ("reg", forest)]).fit(train[features], train[target])
        return regression_metrics(test[target], pipe.predict(test[features]))

    without_apparent_max = [f for f in _ORIGINAL_FEATURES if f != "apparent_temperature_max"]
    without_all_same_day = [f for f in _ORIGINAL_FEATURES if f not in _SAME_DAY_TEMPERATURE]

    setups = [
        ("A", "Original setup: random 80/20 split, all same-day inputs", _ORIGINAL_FEATURES, "random"),
        ("B", "Chronological split (train 2000-2019, test 2020-2024)", _ORIGINAL_FEATURES, "time"),
        ("C", "B, minus apparent_temperature_max", without_apparent_max, "time"),
        ("D", "B, minus all same-day temperature-derived inputs", without_all_same_day, "time"),
    ]
    results = [
        {"id": sid, "setup": label, **fit_score(feats, split)}
        for sid, label, feats, split in setups
    ]

    # City + month average baseline for the same chronological test period.
    train, test = df[time_train], df[~time_train]
    means = train.groupby(["city", train["date"].dt.month])[target].mean()
    keys = pd.MultiIndex.from_arrays([test["city"], test["date"].dt.month])
    results.append(
        {"id": "Baseline", "setup": "City + month average (no machine learning)",
         **regression_metrics(test[target], means.reindex(keys).to_numpy())}
    )
    return results
