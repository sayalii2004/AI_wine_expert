# AI Wine Expert 🍷🥂

A wine quality prediction and recommendation app with a Flask API backend and a React/Vite frontend.

## What it does

- Predicts wine quality score and grade for red or white wine
- Generates SHAP-based explanations for model predictions
- Recommends actionable improvements to move the wine toward a higher grade
- Provides analytics endpoints for feature importance, grade statistics, and model metrics

## Repository structure

- `app.py` - Flask API entrypoint
- `requirements.txt` - Python dependencies for the backend
- `frontend/` - React + Vite user interface
- `src/` - shared Python model, preprocessing, explainability, and recommendation code
- `models/` - trained model artifacts and metadata
- `data/` - wine quality datasets used for training and analysis

## Requirements

- Python 3.11+ (or a compatible Python 3.x version)
- Node.js 18+ / npm 10+ for the frontend

## Local development

### Backend

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install backend dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Flask app:

```powershell
python app.py
```

By default, the backend serves API endpoints under `http://localhost:5000/api`.

### Frontend

1. Change to the frontend folder:

```powershell
cd frontend
```

2. Install dependencies:

```powershell
npm install
```

3. Start the development server:

```powershell
npm run dev
```

4. Open the URL shown by Vite in your browser.

> If the frontend needs to call a separately hosted backend, set `VITE_API_URL` in `frontend/.env`.

## Available backend API endpoints

- `POST /api/predict` — returns `predicted_quality`, `predicted_grade`, and `expected_quality`
- `POST /api/explain` — returns SHAP explanation and feature contributions
- `POST /api/recommend` — returns recommendations for improving the wine
- `GET /api/analytics/global-importance` — returns feature importances
- `GET /api/analytics/grade-stats?wine_type=red&grade=Good` — returns per-grade feature statistics
- `GET /api/analytics/model-metrics` — returns model performance metadata
- `GET /api/health` — health check with model metadata

## Deployment recommendations

### Recommended approach

- Deploy the Flask backend (`app.py`) to a Python-friendly host such as Render, Railway, Fly.io, or Cloud Run.
- Deploy the frontend (`frontend/`) to a static frontend host such as Vercel, Netlify, or Render Static Sites.

### Combined deployment option

If you prefer a single deployable app, use a Docker-based host like Fly.io or Google Cloud Run and serve both backend and built frontend from one container.

### Production notes

- Use a production WSGI server for Flask instead of the development server.
- Restrict CORS origins in `app.py` for production use.

## Notes

- The API uses pre-trained model artifacts from `models/`.
- The frontend calls the backend via the `frontend/src/api.js` client.
- `frontend/package.json` contains the React/Vite build and development scripts.

## Quick start

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
npm run dev
```

Then point your browser at the URL shown by Vite and use the app.
