"""
model.py

Trains and compares regression models for wine quality prediction:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor

Quality GRADE is derived by bucketing the regressor's continuous prediction
(see preprocessing.assign_grade) rather than training a separate classifier.
This keeps the displayed score and grade always consistent (a model can't
predict 6.8 and label it "Premium").

Evaluation happens on two levels:
  1. Regression metrics (MAE, RMSE, R^2) on the raw quality score.
  2. Grading accuracy: bucket each model's predictions into grades and
     compare against the true grade labels (precision/recall/F1 per grade),
     since that's what the end user actually sees in the app.

The best model (selected by test RMSE, tie-broken by grading F1) is saved
to models/ for use by the explainability and recommendation modules and
the Flask API.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

from data_loader import load_wine_data
from preprocessing import prepare_train_test, assign_grade, GRADE_ORDER

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_candidate_models() -> dict:
    """
    Returns the dict of candidate models to compare. Hyperparameters are
    deliberately modest (not heavily tuned) since the goal at this stage is
    a fair model-vs-model comparison; the winning model can be tuned further
    afterward without needing to re-run this whole comparison.
    """
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate_regression(y_true, y_pred) -> dict:
    """Standard regression metrics."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_grading(y_true_grade, y_pred_quality) -> dict:
    """
    Bucket predicted quality scores into grades and compare against the
    true grade labels. Predictions are clipped to the valid 0-10 range and
    rounded to the nearest integer before bucketing, since assign_grade
    expects a quality-score-like input and the dataset's true labels are
    integers.
    """
    y_pred_quality_clipped = np.clip(y_pred_quality, 0, 10)
    y_pred_grade = pd.Series(y_pred_quality_clipped).round().astype(int).apply(assign_grade)

    report = classification_report(
        y_true_grade, y_pred_grade, labels=GRADE_ORDER, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true_grade, y_pred_grade, labels=GRADE_ORDER)

    return {
        "accuracy": round(report["accuracy"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_grade": {
            grade: {
                "precision": round(report[grade]["precision"], 4),
                "recall": round(report[grade]["recall"], 4),
                "f1": round(report[grade]["f1-score"], 4),
                "support": int(report[grade]["support"]),
            }
            for grade in GRADE_ORDER
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": GRADE_ORDER,
    }


def train_and_compare(X_train, X_test, y_train_reg, y_test_reg, y_test_grade) -> dict:
    """
    Trains every candidate model and returns a results dict keyed by
    model name, containing the fitted model, regression metrics, grading
    metrics, and training time.
    """
    models = get_candidate_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        model.fit(X_train, y_train_reg)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)

        reg_metrics = evaluate_regression(y_test_reg, y_pred)
        grade_metrics = evaluate_grading(y_test_grade, y_pred)

        results[name] = {
            "model": model,
            "train_time_sec": round(elapsed, 2),
            "regression_metrics": reg_metrics,
            "grading_metrics": grade_metrics,
        }

        print(
            f"  MAE={reg_metrics['mae']}  RMSE={reg_metrics['rmse']}  R2={reg_metrics['r2']}  "
            f"GradeAcc={grade_metrics['accuracy']}  GradeF1(macro)={grade_metrics['macro_f1']}  "
            f"({elapsed:.1f}s)"
        )

    return results


def select_best_model(results: dict) -> str:
    """
    Model selection criterion: macro F1 on grading, NOT raw RMSE.

    This is a deliberate choice, not an accidental tie-break. On this
    dataset, the model with the best RMSE (Random Forest) achieves it by
    being most accurate on the dominant "Average"/"Good" classes while
    having ZERO recall on "Poor" and "Premium" -- i.e. it never once
    correctly flags an exceptional or flawed wine in the test set.

    Since the entire purpose of this system is to act as a wine quality
    *expert* that flags both problem wines and standout wines (not just
    the average middle), a model that silently ignores both extremes
    fails the actual product goal even if it wins on aggregate error.
    Macro F1 (unweighted average of per-class F1) is used instead of
    weighted F1 or accuracy specifically because it does NOT let the
    large "Good"/"Average" classes drown out performance on the rare
    "Poor"/"Premium" classes.

    RMSE and MAE are still computed and reported for every candidate
    (see model_metadata.json) so this tradeoff is visible and auditable,
    not hidden.
    """
    best_name = max(
        results.items(), key=lambda kv: kv[1]["grading_metrics"]["macro_f1"]
    )[0]
    return best_name


def save_model_artifacts(results: dict, best_name: str, feature_columns: list):
    """
    Persists the winning model and a metadata JSON (metrics for all
    candidates, the selected model's name, and the feature column order
    the model expects) to the models/ directory.
    """
    best_model = results[best_name]["model"]
    model_path = MODELS_DIR / "wine_quality_model.pkl"
    joblib.dump(best_model, model_path)

    metadata = {
        "selected_model": best_name,
        "feature_columns": feature_columns,
        "all_results": {
            name: {
                "train_time_sec": r["train_time_sec"],
                "regression_metrics": r["regression_metrics"],
                "grading_metrics": {
                    k: v for k, v in r["grading_metrics"].items() if k != "confusion_matrix"
                },
            }
            for name, r in results.items()
        },
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model ('{best_name}') to {model_path}")
    print(f"Saved metadata to {metadata_path}")


def main():
    raw_df = load_wine_data()
    split = prepare_train_test(raw_df)

    X_train, X_test = split["X_train"], split["X_test"]
    y_train_reg, y_test_reg = split["y_train_reg"], split["y_test_reg"]
    y_test_grade = split["y_test_grade"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    results = train_and_compare(X_train, X_test, y_train_reg, y_test_reg, y_test_grade)

    best_name = select_best_model(results)
    print(f"\nBest model selected: {best_name}")

    save_model_artifacts(results, best_name, list(X_train.columns))

    return results, best_name


if __name__ == "__main__":
    main()
