"""
recommender.py

Generates actionable improvement suggestions for a wine based on:
  1. SHAP feature attributions (which features are hurting the score)
  2. Grade-banded feature statistics per wine type (what does a wine
     in the next grade up typically look like?)

Design rationale
----------------
A naive approach would rank features by their SHAP contribution and tell
the user to "increase" or "decrease" each one. This fails in practice
because most features in this dataset are NOT monotonically related to
quality — their optimal values depend on wine type and on interactions
with other features. Telling a winemaker to "reduce alcohol" (because a
poor wine happens to have high alcohol) would be misleading when alcohol
is globally the strongest positive predictor of quality.

Instead, recommendations are generated only when THREE conditions are
simultaneously true for a feature:

  Condition 1 — SHAP says it's hurting:
    The feature's SHAP value for this wine is negative (it pulled the
    predicted score below the baseline).

  Condition 2 — The value is actually out of range for the next grade:
    The feature's value sits outside the interquartile range (p25–p75)
    of wines in the target grade, computed per wine type. If the value
    is already inside that IQR, no nudge is needed even if SHAP is
    mildly negative (the feature just isn't the bottleneck).

  Condition 3 — The direction of the nudge is consistent with what
    higher-grade wines look like:
    The suggested adjustment (increase or decrease) must move the value
    toward the target grade's median. This prevents contradictory advice
    (e.g. suggesting "increase volatile acidity" when Premium wines have
    lower volatile acidity than the input wine).

Only features that pass all three conditions are surfaced as suggestions.
Features that pass Condition 1 but fail Condition 2 or 3 are reported as
"already acceptable" so the user understands why they're not being flagged.

Target grade
------------
If the predicted grade is Poor → target Average.
If Average → target Good.
If Good → target Premium.
If already Premium → no improvements suggested (report this clearly).

Feature display names and units are included in output for direct use
by the React frontend without additional mapping.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from data_loader import load_wine_data, FEATURE_COLUMNS
from preprocessing import (
    clean_and_prepare,
    assign_grade,
    GRADE_ORDER,
    WINE_TYPE_MAP,
)
from explainer import WineExplainer, build_input_row, FEATURE_DISPLAY_NAMES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Features to exclude from recommendations entirely.
# wine_type_encoded is an implementation detail, not something a winemaker
# can adjust; density is a consequence of other variables (primarily alcohol
# and sugar), not an independent lever.
EXCLUDE_FROM_RECOMMENDATIONS = {"wine_type_encoded", "density"}

# Human-readable units for each feature, used by the frontend
FEATURE_UNITS = {
    "fixed acidity": "g/L (tartaric acid)",
    "volatile acidity": "g/L (acetic acid)",
    "citric acid": "g/L",
    "residual sugar": "g/L",
    "chlorides": "g/L (sodium chloride)",
    "free sulfur dioxide": "mg/L",
    "total sulfur dioxide": "mg/L",
    "density": "g/cm³",
    "pH": "",
    "sulphates": "g/L (potassium sulphate)",
    "alcohol": "% vol",
    "wine_type_encoded": "",
}

NEXT_GRADE = {
    "Poor": "Average",
    "Average": "Good",
    "Good": "Premium",
    "Premium": None,
}


class WineRecommender:
    """
    Generates improvement recommendations for a wine.

    Intended to be instantiated once at Flask app startup (alongside
    WineExplainer) since it pre-computes and caches grade statistics.
    """

    def __init__(
        self,
        explainer: Optional[WineExplainer] = None,
        metadata_path: Path = MODELS_DIR / "model_metadata.json",
    ):
        # Load or reuse an already-initialised WineExplainer
        self.explainer = explainer or WineExplainer()

        with open(metadata_path) as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]

        # Pre-compute grade statistics (per wine type) at init time
        self._grade_stats = self._build_grade_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, input_df: pd.DataFrame, wine_type: str) -> dict:
        """
        Generate improvement suggestions for a single wine.

        Parameters
        ----------
        input_df  : single-row DataFrame matching model feature columns
        wine_type : "red" or "white"

        Returns
        -------
        dict with keys:
            predicted_quality   : float
            predicted_grade     : str
            target_grade        : str | None
            explanation         : dict  (full SHAP local explanation)
            recommendations     : list[dict]  — actionable suggestions
            already_acceptable  : list[dict]  — negative-SHAP features
                                               already within target IQR
            not_actionable      : list[dict]  — excluded features
                                               (density, wine_type)
        """
        explanation = self.explainer.explain_local(input_df)
        predicted_quality = explanation["predicted_quality"]
        predicted_grade = assign_grade(round(predicted_quality))
        target_grade = NEXT_GRADE[predicted_grade]

        if target_grade is None:
            return {
                "predicted_quality": predicted_quality,
                "predicted_grade": predicted_grade,
                "target_grade": None,
                "explanation": explanation,
                "recommendations": [],
                "already_acceptable": [],
                "not_actionable": [],
                "message": (
                    "This wine is already predicted as Premium — "
                    "no further improvements are suggested."
                ),
            }

        grade_stats = self._grade_stats[wine_type.lower()][target_grade]
        negative_features = explanation["negative_contributors"]

        recommendations = []
        already_acceptable = []
        not_actionable = []

        for entry in negative_features:
            feature = entry["feature"]
            current_value = entry["value"]
            shap_value = entry["shap_value"]

            if feature in EXCLUDE_FROM_RECOMMENDATIONS:
                not_actionable.append({
                    "feature": feature,
                    "display_name": entry["display_name"],
                    "reason": "not directly adjustable",
                    "shap_value": shap_value,
                })
                continue

            stats = grade_stats.get(feature)
            if stats is None:
                continue

            target_median = stats["p50"]
            target_p25 = stats["p25"]
            target_p75 = stats["p75"]

            # Condition 2: is the value already within target IQR?
            in_target_iqr = target_p25 <= current_value <= target_p75

            if in_target_iqr:
                already_acceptable.append({
                    "feature": feature,
                    "display_name": entry["display_name"],
                    "current_value": current_value,
                    "target_iqr": [target_p25, target_p75],
                    "shap_value": shap_value,
                    "note": (
                        f"Already within the typical range for {target_grade} wines "
                        f"({target_p25}–{target_p75})"
                    ),
                })
                continue

            # Condition 3: what direction is needed and does it align
            # with moving toward the target median?
            if current_value < target_p25:
                direction = "increase"
                target_value = target_median
            else:
                direction = "decrease"
                target_value = target_median

            recommendations.append({
                "feature": feature,
                "display_name": FEATURE_DISPLAY_NAMES.get(feature, feature),
                "unit": FEATURE_UNITS.get(feature, ""),
                "current_value": round(current_value, 4),
                "target_value": round(target_value, 4),
                "target_iqr": [round(target_p25, 4), round(target_p75, 4)],
                "direction": direction,
                "shap_value": round(shap_value, 4),
                "impact_rank": len(recommendations) + 1,
                "suggestion": _format_suggestion(
                    FEATURE_DISPLAY_NAMES.get(feature, feature),
                    direction,
                    current_value,
                    target_value,
                    FEATURE_UNITS.get(feature, ""),
                    target_grade,
                    target_p25,
                    target_p75,
                ),
            })

        # Sort recommendations by absolute SHAP impact (most impactful first)
        recommendations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        for i, rec in enumerate(recommendations):
            rec["impact_rank"] = i + 1

        return {
            "predicted_quality": predicted_quality,
            "predicted_grade": predicted_grade,
            "target_grade": target_grade,
            "explanation": explanation,
            "recommendations": recommendations,
            "already_acceptable": already_acceptable,
            "not_actionable": not_actionable,
            "message": _summary_message(
                predicted_grade, target_grade, len(recommendations)
            ),
        }

    def recommend_serialisable(self, input_df: pd.DataFrame, wine_type: str) -> dict:
        """JSON-serialisable version of recommend(). Use in Flask routes."""
        result = self.recommend(input_df, wine_type)
        return _make_serialisable(result)

    # ------------------------------------------------------------------
    # Internal: grade statistics
    # ------------------------------------------------------------------

    def _build_grade_stats(self) -> dict:
        """
        Pre-computes per-wine-type, per-grade, per-feature statistics
        (p25, p50, p75) from the full cleaned training dataset.

        Returns a nested dict:
            stats[wine_type][grade][feature] = {p25, p50, p75, mean, std}
        """
        raw = load_wine_data()
        clean = clean_and_prepare(raw)

        stats = {}
        for wtype in ["red", "white"]:
            stats[wtype] = {}
            type_df = clean[clean.wine_type == wtype]
            for grade in GRADE_ORDER:
                stats[wtype][grade] = {}
                grade_df = type_df[type_df.grade == grade]
                for col in FEATURE_COLUMNS:
                    stats[wtype][grade][col] = {
                        "mean": round(float(grade_df[col].mean()), 4),
                        "std": round(float(grade_df[col].std()), 4),
                        "p25": round(float(grade_df[col].quantile(0.25)), 4),
                        "p50": round(float(grade_df[col].quantile(0.50)), 4),
                        "p75": round(float(grade_df[col].quantile(0.75)), 4),
                    }

        return stats

    def get_grade_stats(self, wine_type: str, grade: str) -> dict:
        """
        Expose grade statistics for the Analytics Dashboard.
        Returns per-feature statistics for a given wine type and grade.
        """
        return self._grade_stats[wine_type.lower()][grade]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_suggestion(
    display_name, direction, current, target, unit, target_grade, p25, p75
) -> str:
    unit_str = f" {unit}" if unit else ""
    return (
        f"{direction.capitalize()} {display_name} from {current:.3f}{unit_str} "
        f"toward {target:.3f}{unit_str}. "
        f"{target_grade} wines typically range {p25:.3f}–{p75:.3f}{unit_str}."
    )


def _summary_message(predicted_grade, target_grade, n_recommendations) -> str:
    if n_recommendations == 0:
        return (
            f"This wine is predicted as {predicted_grade}. "
            f"No specific adjustments were identified to move it toward {target_grade} — "
            f"its negative contributors are either excluded (density, wine type) "
            f"or already within the typical range for {target_grade} wines."
        )
    return (
        f"This wine is predicted as {predicted_grade}. "
        f"{n_recommendations} adjustment(s) were identified that may help "
        f"move it toward {target_grade} quality."
    )


def _make_serialisable(obj):
    """Recursively cast numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))

    print("Initialising WineRecommender (loads explainer + grade stats)...")
    recommender = WineRecommender()
    print("Ready.\n")

    # Test 1: a mediocre red wine (expect Average or Poor, get recommendations)
    mediocre_red = {
        "fixed acidity": 8.5,
        "volatile acidity": 0.72,   # high — known quality detractor
        "citric acid": 0.15,
        "residual sugar": 2.0,
        "chlorides": 0.09,
        "free sulfur dioxide": 10.0,
        "total sulfur dioxide": 45.0,
        "density": 0.9975,
        "pH": 3.45,
        "sulphates": 0.55,
        "alcohol": 9.8,             # low — known quality driver
    }
    input_df = build_input_row(mediocre_red, "red")
    result = recommender.recommend(input_df, "red")

    print(f"=== Mediocre Red Wine ===")
    print(f"Predicted quality : {result['predicted_quality']} ({result['predicted_grade']})")
    print(f"Target grade      : {result['target_grade']}")
    print(f"Message           : {result['message']}")
    print()
    print(f"Recommendations ({len(result['recommendations'])}):")
    for rec in result["recommendations"]:
        print(f"  [{rec['impact_rank']}] {rec['suggestion']}")
    print()
    print(f"Already acceptable ({len(result['already_acceptable'])}):")
    for a in result["already_acceptable"]:
        print(f"  {a['display_name']}: {a['note']}")

    print()
    print("---")

    # Test 2: a good wine (expect Good, fewer/no recommendations)
    good_white = {
        "fixed acidity": 6.8,
        "volatile acidity": 0.25,
        "citric acid": 0.35,
        "residual sugar": 3.5,
        "chlorides": 0.04,
        "free sulfur dioxide": 38.0,
        "total sulfur dioxide": 115.0,
        "density": 0.992,
        "pH": 3.22,
        "sulphates": 0.50,
        "alcohol": 12.0,
    }
    input_df2 = build_input_row(good_white, "white")
    result2 = recommender.recommend(input_df2, "white")

    print(f"=== Good White Wine ===")
    print(f"Predicted quality : {result2['predicted_quality']} ({result2['predicted_grade']})")
    print(f"Target grade      : {result2['target_grade']}")
    print(f"Message           : {result2['message']}")
    print()
    print(f"Recommendations ({len(result2['recommendations'])}):")
    for rec in result2["recommendations"]:
        print(f"  [{rec['impact_rank']}] {rec['suggestion']}")