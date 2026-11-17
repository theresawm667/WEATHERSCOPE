"""The three regression models, plus helpers to save and load them safely.

Every model is a scikit-learn Pipeline:
    preprocessing (encode city, optionally scale numbers) -> regressor
Bundling both steps in one object means the app can never forget to apply the
same preprocessing that was used in training.
"""

from __future__ import annotations

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from paths import MODEL_DIR

RANDOM_STATE = 42

MODEL_LABELS = {
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting (Hist)",
}


class ModelError(Exception):
    """Raised with a human-readable message when a model cannot be loaded."""


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode the city; optionally standardise the numeric columns."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", numeric_step, NUMERIC_FEATURES),
        ]
    )


def build_model(key: str) -> Pipeline:
    """Create an UNTRAINED pipeline for the given model key."""
    if key == "ridge":
        # Linear model: needs scaled inputs so no feature dominates by its units.
        return Pipeline([("prep", _preprocessor(True)), ("reg", Ridge(alpha=1.0))])

    if key == "random_forest":
        # Limits on depth / leaf size keep the saved file small and reduce
        forest = RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=30,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", _preprocessor(False)), ("reg", forest)])

    if key == "hist_gradient_boosting":
        booster = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_depth=6,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
        return Pipeline([("prep", _preprocessor(False)), ("reg", booster)])

    raise ValueError(f"Unknown model key: {key!r}. Choose from {list(MODEL_LABELS)}.")


def model_path(key: str):
    return MODEL_DIR / f"{key}.joblib"


def save_model(model: Pipeline, key: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(key)
    joblib.dump(model, path, compress=3)
    return path


def load_model(key: str) -> Pipeline:
    """Load a trained model, with clear instructions if that fails."""
    path = model_path(key)
    if not path.exists():
        raise ModelError(
            f"Trained model '{MODEL_LABELS.get(key, key)}' not found "
            f"({path.name}).\n\nRun this in the project folder to create it:\n"
            "    python train.py"
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # most often: scikit-learn version mismatch
        raise ModelError(
            f"The saved model {path.name} could not be loaded ({exc}).\n\n"
            "This usually means it was trained with a different scikit-learn "
            "version. Fix it by retraining on this computer:\n"
            "    python train.py"
        ) from exc
