"""
app.py — AI Wine Expert Flask API

Endpoints
---------
POST /api/predict
    Input:  { features: {...11 fields...}, wine_type: "red"|"white" }
    Output: { predicted_quality, predicted_grade, expected_quality }

POST /api/explain
    Input:  { features: {...}, wine_type: "red"|"white" }
    Output: { predicted_quality, expected_quality, shap_values,
              positive_contributors, negative_contributors }

POST /api/recommend
    Input:  { features: {...}, wine_type: "red"|"white" }
    Output: { predicted_quality, predicted_grade, target_grade,
              explanation, recommendations, already_acceptable,
              not_actionable, message }

GET  /api/analytics/global-importance
    Output: { feature_importances: [{feature, display_name, mean_abs_shap}] }

GET  /api/analytics/grade-stats?wine_type=red&grade=Good
    Output: { wine_type, grade, stats: {feature: {mean,std,p25,p50,p75}} }

GET  /api/analytics/model-metrics
    Output: { selected_model, all_results: {...} }

GET  /api/health
    Output: { status: "ok", model: "lightgbm" }

Design notes
------------
- WineExplainer and WineRecommender are instantiated ONCE at startup
  (not per-request) since TreeExplainer initialisation takes ~0.5s and
  the background dataset / grade stats are static.

- All POST endpoints share the same input validation (validate_features).
  A 400 with a clear message is returned for any missing or non-numeric
  field, so the React frontend gets actionable error text rather than a
  500 stack trace.

- CORS is enabled for all origins in development. For production on
  Render/Vercel, restrict CORS_ORIGINS in config to your Vercel domain.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add src/ to path so relative imports work when running from project root
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import FEATURE_COLUMNS
from preprocessing import assign_grade, GRADE_ORDER, WINE_TYPE_MAP
from explainer import WineExplainer, build_input_row
from recommender import WineRecommender

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # restrict to CORS_ORIGINS env var in production if needed

MODELS_DIR = Path(__file__).resolve().parent / "models"

# ---------------------------------------------------------------------------
# Startup: load models once
# ---------------------------------------------------------------------------

print("Loading WineExplainer...")
_explainer = WineExplainer(
    model_path=MODELS_DIR / "wine_quality_model.pkl",
    background_path=MODELS_DIR / "shap_background.pkl",
    metadata_path=MODELS_DIR / "model_metadata.json",
)
print(f"  Model loaded. Baseline quality: {_explainer.expected_value:.3f}")

print("Loading WineRecommender (pre-computing grade stats)...")
_recommender = WineRecommender(
    explainer=_explainer,
    metadata_path=MODELS_DIR / "model_metadata.json",
)
print("  Ready.\n")

with open(MODELS_DIR / "model_metadata.json") as f:
    _metadata = json.load(f)

# Cache global importance at startup (static, no need to recompute per request)
_global_importance = _explainer.explain_global_serialisable()

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_request(data: dict) -> tuple[dict | None, str | None]:
    """
    Validates a prediction/explain/recommend request body.
    Returns (parsed_dict, None) on success or (None, error_message) on failure.
    """
    if not data:
        return None, "Request body is missing or not valid JSON."

    wine_type = data.get("wine_type", "").lower()
    if wine_type not in WINE_TYPE_MAP:
        return None, f"wine_type must be 'red' or 'white', got: '{wine_type}'"

    features_raw = data.get("features")
    if not isinstance(features_raw, dict):
        return None, "Request body must contain a 'features' object."

    features = {}
    missing = []
    invalid = []

    for col in FEATURE_COLUMNS:
        if col == "wine_type_encoded":
            continue  # handled separately
        val = features_raw.get(col)
        if val is None:
            missing.append(col)
        else:
            try:
                features[col] = float(val)
            except (TypeError, ValueError):
                invalid.append(col)

    if missing:
        return None, f"Missing feature(s): {', '.join(missing)}"
    if invalid:
        return None, f"Non-numeric value(s) for feature(s): {', '.join(invalid)}"

    return {"features": features, "wine_type": wine_type}, None


def handle_errors(f):
    """Decorator: catches unexpected exceptions and returns a 500 JSON response."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(traceback.format_exc())
            return jsonify({"error": "Internal server error", "detail": str(e)}), 500
    return wrapper


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model": _metadata.get("selected_model", "unknown"),
        "baseline_quality": round(_explainer.expected_value, 3),
    })


@app.post("/api/predict")
@handle_errors
def predict():
    """
    Lightweight prediction endpoint — returns score + grade only.
    Use /api/recommend if you also need SHAP values and suggestions.
    """
    parsed, err = validate_request(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    input_df = build_input_row(parsed["features"], parsed["wine_type"])
    predicted_quality = float(_explainer.model.predict(input_df)[0])
    predicted_grade = assign_grade(round(predicted_quality))

    return jsonify({
        "predicted_quality": round(predicted_quality, 2),
        "predicted_grade": predicted_grade,
        "expected_quality": round(_explainer.expected_value, 2),
    })


@app.post("/api/explain")
@handle_errors
def explain():
    """
    Full SHAP explanation for a single wine.
    Returns positive/negative contributors with per-feature SHAP values.
    """
    parsed, err = validate_request(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    input_df = build_input_row(parsed["features"], parsed["wine_type"])
    result = _explainer.explain_local_serialisable(input_df)
    result["predicted_grade"] = assign_grade(round(result["predicted_quality"]))
    return jsonify(result)


@app.post("/api/recommend")
@handle_errors
def recommend():
    """
    Full recommendation response: SHAP explanation + improvement suggestions.
    This is the primary endpoint for the app's main prediction panel.
    """
    parsed, err = validate_request(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    input_df = build_input_row(parsed["features"], parsed["wine_type"])
    result = _recommender.recommend_serialisable(input_df, parsed["wine_type"])

    # Drop the nested explanation.shap_values from the top-level to avoid
    # bloating the response — the frontend only needs the summary fields
    # from the explanation at the top level
    if "explanation" in result:
        result["shap_values"] = result["explanation"].get("shap_values", [])
        result["positive_contributors"] = result["explanation"].get("positive_contributors", [])
        result["negative_contributors"] = result["explanation"].get("negative_contributors", [])
        result["expected_quality"] = result["explanation"].get("expected_quality")
        del result["explanation"]

    return jsonify(result)


@app.get("/api/analytics/global-importance")
@handle_errors
def global_importance():
    """
    Global feature importances (mean |SHAP| over background dataset).
    Static — computed once at startup. Powers the Feature Importance chart.
    """
    return jsonify(_global_importance)


@app.get("/api/analytics/grade-stats")
@handle_errors
def grade_stats():
    """
    Per-grade feature statistics for a given wine type.
    Query params: wine_type=red|white, grade=Poor|Average|Good|Premium
    Powers the grade distribution charts on the Analytics Dashboard.
    """
    wine_type = request.args.get("wine_type", "").lower()
    grade = request.args.get("grade", "")

    if wine_type not in WINE_TYPE_MAP:
        return jsonify({"error": "wine_type must be 'red' or 'white'"}), 400
    if grade not in GRADE_ORDER:
        return jsonify({"error": f"grade must be one of: {', '.join(GRADE_ORDER)}"}), 400

    stats = _recommender.get_grade_stats(wine_type, grade)
    return jsonify({"wine_type": wine_type, "grade": grade, "stats": stats})


@app.get("/api/analytics/model-metrics")
@handle_errors
def model_metrics():
    """
    Full model comparison metrics from training.
    Powers the Model Performance Comparison panel on the Analytics Dashboard.
    """
    return jsonify(_metadata)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)