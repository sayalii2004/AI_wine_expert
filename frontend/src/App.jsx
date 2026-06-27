import React, { useState } from "react";
import "./index.css";
import InputPanel, { DEFAULT_FEATURES } from "./components/InputPanel";
import QualityDial from "./components/QualityDial";
import ExplanationPanel from "./components/ExplanationPanel";
import RecommendationPanel from "./components/RecommendationPanel";
import AnalyticsPanel from "./components/AnalyticsPanel";
import { recommend } from "./api";

const TABS = ["Prediction", "Explanation", "Recommendations", "Analytics"];

const TAB_ICON = {
  Prediction: "🍷",
  Explanation: "🔬",
  Recommendations: "💡",
  Analytics: "📊",
};

export default function App() {
  const [features, setFeatures] = useState({ ...DEFAULT_FEATURES });
  const [wineType, setWineType] = useState("red");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("Prediction");

  const handleChange = (key, val) =>
    setFeatures((prev) => ({ ...prev, [key]: val }));

  const handleAnalyse = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await recommend(features, wineType);
      setResult(data);
      setActiveTab("Prediction");
    } catch (e) {
      setError(
        e.response?.data?.error ||
          "Could not reach the API. Is Flask running on port 5000?",
      );
    } finally {
      setLoading(false);
    }
  };

  const score = result?.predicted_quality ?? null;
  const grade = result?.predicted_grade ?? null;

  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        background: "var(--parchment)",
      }}
    >
      {/* Left sidebar */}
      <InputPanel
        features={features}
        wineType={wineType}
        onChange={handleChange}
        onWineTypeChange={setWineType}
        onAnalyse={handleAnalyse}
        loading={loading}
      />

      {/* Right main panel */}
      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        {/* Top bar */}
        <div
          style={{
            background: "white",
            borderBottom: "1px solid var(--border)",
            padding: "16px 32px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            position: "sticky",
            top: 0,
            zIndex: 10,
          }}
        >
          <div>
            <div
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 20,
                fontWeight: 700,
                color: "var(--burgundy)",
              }}
            >
              Wine Quality Analysis
            </div>
            <div
              style={{ fontSize: 11, color: "var(--slate-lt)", marginTop: 1 }}
            >
              {result
                ? `${wineType.charAt(0).toUpperCase() + wineType.slice(1)} wine · Analysed`
                : "Adjust sliders and click Analyse Wine"}
            </div>
          </div>

          {/* Dial lives in the top bar when results are present */}
          <QualityDial score={score} grade={grade} loading={loading} />
        </div>

        {/* Error banner */}
        {error && (
          <div
            style={{
              margin: "16px 32px 0",
              padding: "12px 16px",
              background: "var(--negative-lt)",
              border: "1px solid var(--negative)",
              borderRadius: 8,
              color: "var(--negative)",
              fontSize: 13,
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: 0,
            padding: "0 32px",
            borderBottom: "1px solid var(--border)",
            background: "white",
            marginTop: 0,
          }}
        >
          {TABS.map((tab) => {
            const active = tab === activeTab;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: "14px 20px",
                  background: "none",
                  border: "none",
                  borderBottom: active
                    ? "3px solid var(--burgundy)"
                    : "3px solid transparent",
                  color: active ? "var(--burgundy)" : "var(--slate-lt)",
                  fontWeight: active ? 700 : 400,
                  fontSize: 13,
                  cursor: "pointer",
                  fontFamily: "var(--font-body)",
                  transition: "all 0.15s",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <span>{TAB_ICON[tab]}</span> {tab}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <div style={{ flex: 1, padding: "28px 32px", maxWidth: 820 }}>
          {activeTab === "Prediction" && (
            <PredictionTab
              result={result}
              loading={loading}
              wineType={wineType}
            />
          )}
          {activeTab === "Explanation" && (
            <ExplanationPanel
              data={
                result
                  ? {
                      predicted_quality: result.predicted_quality,
                      expected_quality: result.expected_quality,
                      predicted_grade: result.predicted_grade,
                      shap_values: result.shap_values,
                    }
                  : null
              }
            />
          )}
          {activeTab === "Recommendations" && (
            <RecommendationPanel data={result} />
          )}
          {activeTab === "Analytics" && <AnalyticsPanel />}
        </div>
      </main>
    </div>
  );
}

// ─── Prediction tab ────────────────────────────────────────────────────────

const GRADE_COLOR = {
  Poor: "#C0392B",
  Average: "#E67E22",
  Good: "#2D6A4F",
  Premium: "#B8860B",
};

function PredictionTab({ result, loading, wineType }) {
  if (loading) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "80px 0",
          color: "var(--slate-lt)",
        }}
      >
        <div style={{ fontSize: 40, marginBottom: 16 }}>🍷</div>
        <div style={{ fontFamily: "var(--font-display)", fontSize: 18 }}>
          Analysing wine chemistry…
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "80px 0",
          color: "var(--slate-lt)",
        }}
      >
        <div style={{ fontSize: 48, marginBottom: 16 }}>🍾</div>
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 22,
            color: "var(--burgundy)",
            marginBottom: 8,
          }}
        >
          Ready to assess your wine
        </div>
        <div
          style={{
            fontSize: 14,
            lineHeight: 1.7,
            maxWidth: 420,
            margin: "0 auto",
          }}
        >
          Adjust the physicochemical properties in the left panel, then click{" "}
          <strong>Analyse Wine</strong> to get a quality score, grade, and
          improvement suggestions.
        </div>
      </div>
    );
  }

  const {
    predicted_quality,
    predicted_grade,
    target_grade,
    message,
    positive_contributors,
    negative_contributors,
  } = result;
  const gradeColor = GRADE_COLOR[predicted_grade];

  return (
    <div>
      {/* Score card */}
      <div
        style={{
          background: "white",
          borderRadius: 12,
          padding: "24px 28px",
          boxShadow: "var(--shadow)",
          border: "1px solid var(--border)",
          marginBottom: 20,
          display: "flex",
          gap: 32,
          alignItems: "center",
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "var(--slate-lt)",
              marginBottom: 6,
            }}
          >
            Predicted Quality Score
          </div>
          <div
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 48,
              fontWeight: 700,
              color: gradeColor,
              lineHeight: 1,
            }}
          >
            {predicted_quality.toFixed(1)}
            <span
              style={{
                fontSize: 18,
                color: "var(--slate-lt)",
                fontWeight: 400,
                marginLeft: 6,
              }}
            >
              /10
            </span>
          </div>
          <div
            style={{
              marginTop: 10,
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}
          >
            <span
              style={{
                background: gradeColor,
                color: "white",
                fontFamily: "var(--font-display)",
                fontSize: 14,
                fontWeight: 700,
                padding: "4px 16px",
                borderRadius: 20,
              }}
            >
              {predicted_grade}
            </span>
            {target_grade && (
              <span style={{ fontSize: 12, color: "var(--slate-lt)" }}>
                → Target:{" "}
                <strong style={{ color: GRADE_COLOR[target_grade] }}>
                  {target_grade}
                </strong>
              </span>
            )}
          </div>
        </div>

        <div
          style={{
            background: "var(--parchment)",
            borderRadius: 10,
            padding: "14px 20px",
            fontSize: 13,
            color: "var(--slate)",
            lineHeight: 1.6,
            maxWidth: 280,
            borderLeft: `4px solid ${gradeColor}`,
          }}
        >
          {message}
        </div>
      </div>

      {/* Top contributors summary */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
          marginBottom: 20,
        }}
      >
        <ContributorCard
          title="Top Positive Factors"
          icon="↑"
          color="var(--positive)"
          bg="var(--positive-lt)"
          items={positive_contributors?.slice(0, 3)}
        />
        <ContributorCard
          title="Top Negative Factors"
          icon="↓"
          color="var(--negative)"
          bg="var(--negative-lt)"
          items={negative_contributors?.slice(0, 3)}
        />
      </div>

      {/* Quick nav hint */}
      <div
        style={{
          fontSize: 12,
          color: "var(--slate-lt)",
          textAlign: "center",
          padding: "8px 0",
        }}
      >
        See the <strong>Explanation</strong> tab for full SHAP analysis ·{" "}
        <strong>Recommendations</strong> for improvement suggestions
      </div>
    </div>
  );
}

function ContributorCard({ title, icon, color, bg, items }) {
  return (
    <div
      style={{
        background: "white",
        borderRadius: 10,
        padding: "16px",
        boxShadow: "var(--shadow)",
        border: "1px solid var(--border)",
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontWeight: 700,
          color,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 12,
        }}
      >
        {icon} {title}
      </div>
      {items?.length ? (
        items.map((item) => (
          <div
            key={item.feature}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 8,
            }}
          >
            <div>
              <div
                style={{ fontSize: 13, fontWeight: 500, color: "var(--slate)" }}
              >
                {item.display_name}
              </div>
              <div style={{ fontSize: 11, color: "var(--slate-lt)" }}>
                value: {item.value}
              </div>
            </div>
            <div
              style={{
                background: bg,
                color,
                fontWeight: 700,
                fontSize: 12,
                padding: "2px 10px",
                borderRadius: 12,
              }}
            >
              {item.shap_value > 0 ? "+" : ""}
              {item.shap_value.toFixed(3)}
            </div>
          </div>
        ))
      ) : (
        <div style={{ fontSize: 12, color: "var(--slate-lt)" }}>None</div>
      )}
    </div>
  );
}
