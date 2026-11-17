"""Train all models and write the results the app displays.

Run from the project folder:
    python train.py

Outputs (all regenerated every run):
    models/*.joblib              trained models
    reports/metrics.json         every number shown in the "Model Lab" page
    reports/test_predictions.csv per-day test-period predictions (for charts)

The procedure, in plain English:
  1. Fit each model on 2000-2019, score it on 2020-2021 (validation).
     -> used to COMPARE models and pick the best one.
  2. Re-fit each model on 2000-2021, score it ONCE on 2022-2024 (test).
     -> the honest final numbers; the saved model is this re-fitted one.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import sklearn

from data import DataError, load_weather_data, validate_weather_data
from evaluate import (
    TRAIN_END, VALIDATION_END, climatology_predictions, error_breakdown,
    importance_table, leakage_demo, persistence_predictions,
    regression_metrics, split_by_target_date,
)
from features import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from models import MODEL_LABELS, build_model, save_model
from paths import METRICS_PATH, MODEL_DIR, PREDICTIONS_PATH, REPORT_DIR


def log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def main() -> int:
    started = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    try:
        raw = load_weather_data()
    except DataError as exc:
        log(f"ERROR: {exc}")
        return 1
    for warning in validate_weather_data(raw):
        log(f"warning: {warning}")

    feats = build_features(raw, require_target=True)
    masks = split_by_target_date(feats)
    train, val, test = feats[masks["train"]], feats[masks["validation"]], feats[masks["test"]]
    train_val = feats[masks["train"] | masks["validation"]]
    log(f"rows -> train {len(train):,} | validation {len(val):,} | test {len(test):,}")

    # 2. Baselines ----------------------------------------------------------
    baselines = {
        "persistence": {
            "label": "Baseline: tomorrow = today",
            "validation": regression_metrics(val[TARGET_COLUMN], persistence_predictions(val)),
            "test": regression_metrics(test[TARGET_COLUMN], persistence_predictions(test)),
        },
        "climatology": {
            "label": "Baseline: city + month average",
            # Averages come from the data BEFORE the period being scored.
            "validation": regression_metrics(val[TARGET_COLUMN], climatology_predictions(train, val)),
            "test": regression_metrics(test[TARGET_COLUMN], climatology_predictions(train_val, test)),
        },
    }

    # 3. Models -------------------------------------------------------------
    results: dict[str, dict] = {}
    test_frame = test[["city", "date", "target_date", TARGET_COLUMN]].rename(
        columns={TARGET_COLUMN: "actual"}
    )
    test_frame["persistence"] = persistence_predictions(test)
    test_frame["climatology"] = climatology_predictions(train_val, test)
    final_models = {}

    for key, label in MODEL_LABELS.items():
        t0 = time.time()
        # (a) fit on train, score on validation -> for choosing between models
        model = build_model(key).fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        val_metrics = regression_metrics(val[TARGET_COLUMN], model.predict(val[FEATURE_COLUMNS]))

        # (b) re-fit on train + validation, score once on the untouched test set
        final = build_model(key).fit(train_val[FEATURE_COLUMNS], train_val[TARGET_COLUMN])
        pred = final.predict(test[FEATURE_COLUMNS])
        test_metrics = regression_metrics(test[TARGET_COLUMN], pred)
        residual = test[TARGET_COLUMN].to_numpy() - pred

        frame = pd.DataFrame(
            {"city": test["city"].to_numpy(),
             "month": test["target_date"].dt.month.to_numpy(),
             "y_true": test[TARGET_COLUMN].to_numpy(), "y_pred": pred}
        )
        results[key] = {
            "label": label,
            "validation": val_metrics,
            "test": test_metrics,
            # 90% of test errors (actual - predicted) fell between these values
            "residual_quantiles": {
                "q05": float(pd.Series(residual).quantile(0.05)),
                "q50": float(pd.Series(residual).quantile(0.50)),
                "q95": float(pd.Series(residual).quantile(0.95)),
            },
            "by_city": error_breakdown(frame, "city").to_dict("records"),
            "by_month": error_breakdown(frame, "month").to_dict("records"),
        }
        test_frame[key] = pred
        final_models[key] = final
        path = save_model(final, key)
        log(f"{label:26s} val MAE {val_metrics['mae']:.2f} | test MAE {test_metrics['mae']:.2f} "
            f"| {path.stat().st_size / 1e6:.1f} MB | {time.time() - t0:.0f}s")

    # 4. Pick the best model by VALIDATION error (never by test error) ------
    best = min(results, key=lambda k: results[k]["validation"]["rmse"])
    log(f"best model on validation RMSE: {MODEL_LABELS[best]}")

    # 5. Feature importance on the test set ---------------------------------
    importance = importance_table(final_models[best], test, test[TARGET_COLUMN])

    # 6. Leakage demonstration ----------------------------------------------
    log("running the leakage demonstration (a few quick Random Forests)...")
    demo = leakage_demo(raw)

    # 7. Write outputs -------------------------------------------------------
    test_frame.to_csv(PREDICTIONS_PATH, index=False, float_format="%.3f")
    report = {
        "meta": {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "train_period": f"{feats['target_date'].min().date()} to {TRAIN_END.date()}",
            "validation_period": f"{(TRAIN_END + pd.Timedelta(days=1)).date()} to {VALIDATION_END.date()}",
            "test_period": f"{(VALIDATION_END + pd.Timedelta(days=1)).date()} to {feats['target_date'].max().date()}",
            "n_train": len(train), "n_validation": len(val), "n_test": len(test),
        },
        "feature_columns": FEATURE_COLUMNS,
        "models": results,
        "baselines": baselines,
        "best_model": best,
        "feature_importance": importance.to_dict("records"),
        "leakage_demo": demo,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {METRICS_PATH.relative_to(METRICS_PATH.parent.parent)}")
    log(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
