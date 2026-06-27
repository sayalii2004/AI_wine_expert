"""
explainer.py

Wraps the trained LightGBM model with SHAP TreeExplainer to produce
per-prediction feature attributions for the Flask API and frontend.

Two kinds of output are provided:

  1. LOCAL explanation  — for a single wine input, which features pushed
     the predicted score up or down, and by how much. This powers the
     "Positive/Negative Contributors" panel in the app.

  2. GLOBAL explanation — feature importance averaged over the full
     background dataset. This powers the "Feature Importance" chart
     on the Analytics Dashboard.

Design notes
------------
- TreeExplainer is used (not KernelExplainer) because the deployed model
  is a tree ensemble (LightGBM). TreeExplainer is exact and fast — O(TLD)
  where T=trees, L=leaves, D=depth — vs. KernelExplainer which is a slow
  model-agnostic approximation. For LightGBM with 300 trees a single
  local explanation takes ~1ms vs. ~200ms for KernelExplainer.

- feature_perturbation='interventional' with a background dataset is used
  rather than the default 'tree_path_dependent'. Interventional SHAP
  breaks correlations between features when computing counterfactuals,
  which produces more trustworthy attributions when features are correlated
  (density and alcohol are moderately correlated in this dataset, r≈-0.5).

- wine_type_encoded is included in SHAP values (it's a real model feature)
  but is relabelled as 'wine_type' in output for frontend readability, since
  the encoded 0/1 meaning is an implementation detail the user shouldn't see.

- The explainer object is heavy to initialise (~0.5s with 200 background
  samples) so WineExplainer is designed to be instantiated once at Flask
  app startup and reused across requests.
"""

from pathlib import Path
from typing import Optional
import json

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Maps internal column names to human-readable display labels
FEATURE_DISPLAY_NAMES = {
    "fixed acidity": "Fixed Acidity",
    "volatile acidity": "Volatile Acidity",
    "citric acid": "Citric Acid",
    "residual sugar": "Residual Sugar",
    "chlorides": "Chlorides",
    "free sulfur dioxide": "Free Sulfur Dioxide",
    "total sulfur dioxide": "Total Sulfur Dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol",
    "wine_type_encoded": "Wine Type",
}


class WineExplainer:
    """
    Loads the trained model + background dataset and exposes methods for
    generating local (per-prediction) and global (dataset-level) SHAP
    explanations. Intended to be instantiated once at app startup.
    """

    def __init__(
        self,
        model_path: Path = MODELS_DIR / "wine_quality_model.pkl",
        background_path: Path = MODELS_DIR / "shap_background.pkl",
        metadata_path: Path = MODELS_DIR / "model_metadata.json",
    ):
        self.model = joblib.load(model_path)
        self.background = joblib.load(background_path)

        with open(metadata_path) as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]

        # shap.TreeExplainer's internal masker defaults to max_samples=100.
        # Passing exactly 100 background rows avoids a noisy subsampling
        # warning while still giving stable expected_value estimation.
        background_100 = (
            self.background.sample(n=100, random_state=42)
            if len(self.background) > 100
            else self.background
        )
        self.explainer = shap.TreeExplainer(
            self.model,
            data=background_100,
            feature_perturbation="interventional",
        )
        self.expected_value = float(self.explainer.expected_value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_local(self, input_df: pd.DataFrame) -> dict:
        """
        Compute a local SHAP explanation for a single wine sample.

        Parameters
        ----------
        input_df : pd.DataFrame
            Single-row DataFrame with the same columns as self.feature_columns.
            Typically produced by preprocessing.build_input_row().

        Returns
        -------
        dict with keys:
            predicted_quality  : float — raw regressor output
            expected_quality   : float — SHAP baseline (dataset mean)
            shap_values        : list[dict] — one entry per feature, sorted by
                                 absolute contribution descending
            positive_contributors : list[dict] — features that raised the score
            negative_contributors : list[dict] — features that lowered the score
        """
        input_df = input_df[self.feature_columns]
        predicted_quality = float(self.model.predict(input_df)[0])
        shap_vals = self.explainer.shap_values(input_df)[0]  # shape: (n_features,)

        entries = []
        for col, sv, val in zip(self.feature_columns, shap_vals, input_df.iloc[0]):
            entries.append({
                "feature": col,
                "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                "value": round(float(val), 4),
                "shap_value": round(float(sv), 4),
                "direction": "positive" if sv > 0 else "negative",
            })

        entries.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "predicted_quality": round(predicted_quality, 2),
            "expected_quality": round(self.expected_value, 2),
            "shap_values": entries,
            "positive_contributors": [e for e in entries if e["shap_value"] > 0],
            "negative_contributors": [e for e in entries if e["shap_value"] < 0],
        }

    def explain_global(self) -> dict:
        """
        Compute global feature importance as mean(|SHAP|) over the
        background dataset. Used for the Analytics Dashboard's feature
        importance chart.

        Returns
        -------
        dict with keys:
            feature_importances : list[dict] sorted by mean_abs_shap descending
        """
        shap_vals = self.explainer.shap_values(self.background)  # (n_samples, n_features)
        mean_abs = np.abs(shap_vals).mean(axis=0)

        importances = []
        for col, importance in zip(self.feature_columns, mean_abs):
            importances.append({
                "feature": col,
                "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
                "mean_abs_shap": round(float(importance), 4),
            })

        importances.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        return {"feature_importances": importances}

    def explain_local_serialisable(self, input_df: pd.DataFrame) -> dict:
        """
        Same as explain_local but guaranteed JSON-serialisable
        (all numpy floats cast to Python floats). Use this in Flask routes.
        """
        result = self.explain_local(input_df)
        return _make_serialisable(result)

    def explain_global_serialisable(self) -> dict:
        """JSON-serialisable version of explain_global. Use in Flask routes."""
        result = self.explain_global()
        return _make_serialisable(result)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_serialisable(obj):
    """Recursively cast numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def build_input_row(features: dict, wine_type: str) -> pd.DataFrame:
    """
    Convenience function for Flask routes: converts a flat dict of
    physicochemical features + a wine_type string into a single-row
    DataFrame in the exact column order the model expects.

    Parameters
    ----------
    features : dict
        Keys matching FEATURE_COLUMNS (excluding wine_type_encoded).
        Example: {"fixed acidity": 7.5, "volatile acidity": 0.4, ...}
    wine_type : str
        "red" or "white"

    Returns
    -------
    pd.DataFrame with one row and columns matching model feature_columns
    """
    from preprocessing import WINE_TYPE_MAP, FEATURE_COLUMNS

    row = {col: features[col] for col in FEATURE_COLUMNS}
    row["wine_type_encoded"] = WINE_TYPE_MAP[wine_type.lower()]
    return pd.DataFrame([row])


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))

    print("Initialising WineExplainer...")
    wx = WineExplainer()
    print(f"  Expected value (baseline quality): {wx.expected_value:.3f}")
    print(f"  Feature columns: {wx.feature_columns}")
    print()

    # Local explanation on a sample wine
    sample_features = {
        "fixed acidity": 7.5,
        "volatile acidity": 0.55,
        "citric acid": 0.25,
        "residual sugar": 2.0,
        "chlorides": 0.07,
        "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 40.0,
        "density": 0.9965,
        "pH": 3.35,
        "sulphates": 0.62,
        "alcohol": 10.5,
    }
    input_df = build_input_row(sample_features, wine_type="red")
    local = wx.explain_local(input_df)

    print(f"Predicted quality : {local['predicted_quality']}")
    print(f"Baseline (expected): {local['expected_quality']}")
    print()
    print("Top contributors:")
    for e in local["shap_values"][:5]:
        sign = "+" if e["shap_value"] > 0 else ""
        print(f"  {e['display_name']:<25} value={e['value']}  SHAP={sign}{e['shap_value']}")

    print()
    print("Positive contributors:", [e["display_name"] for e in local["positive_contributors"]])
    print("Negative contributors:", [e["display_name"] for e in local["negative_contributors"]])

    # Global importance
    print()
    print("Global feature importances (mean |SHAP| over background):")
    global_exp = wx.explain_global()
    for item in global_exp["feature_importances"]:
        bar = "█" * int(item["mean_abs_shap"] * 80)
        print(f"  {item['display_name']:<25} {item['mean_abs_shap']:.4f}  {bar}")