import React from 'react';

const DIRECTION_ICON = { increase: '↑', decrease: '↓' };
const DIRECTION_COLOR = { increase: 'var(--positive)', decrease: 'var(--negative)' };

export default function RecommendationPanel({ data }) {
  if (!data) return <Placeholder />;

  const { predicted_grade, target_grade, recommendations, already_acceptable, message } = data;

  return (
    <div style={{ padding: '8px 0' }}>
      <div style={headingStyle}>Improvement Recommendations</div>

      {/* Grade transition header */}
      {target_grade && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, margin: '12px 0 20px',
          padding: '12px 16px', background: 'var(--blush)',
          borderRadius: 8, borderLeft: '4px solid var(--burgundy-mid)',
        }}>
          <GradeBadge grade={predicted_grade} />
          <span style={{ color: 'var(--slate-lt)', fontSize: 18 }}>→</span>
          <GradeBadge grade={target_grade} dimmed />
          <span style={{ fontSize: 12, color: 'var(--slate-lt)', marginLeft: 4 }}>
            {recommendations?.length
              ? `${recommendations.length} adjustment${recommendations.length > 1 ? 's' : ''} identified`
              : 'Features already in range'}
          </span>
        </div>
      )}

      {/* Premium message */}
      {!target_grade && (
        <div style={{
          padding: '16px', background: '#FFF3CD', borderRadius: 8,
          borderLeft: '4px solid var(--gold)', marginBottom: 20,
        }}>
          <div style={{ fontSize: 13, color: 'var(--slate)' }}>🏆 {message}</div>
        </div>
      )}

      {/* Recommendation cards */}
      {recommendations?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 24 }}>
          {recommendations.map((rec, i) => (
            <RecCard key={rec.feature} rec={rec} rank={i + 1} />
          ))}
        </div>
      )}

      {/* No recommendations message */}
      {recommendations?.length === 0 && target_grade && (
        <div style={{
          padding: '16px', background: 'var(--positive-lt)', borderRadius: 8,
          borderLeft: '4px solid var(--positive)', marginBottom: 20,
          fontSize: 13, color: 'var(--slate)',
        }}>
          ✓ {message}
        </div>
      )}

      {/* Already acceptable */}
      {already_acceptable?.length > 0 && (
        <div>
          <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em',
            color: 'var(--slate-lt)', marginBottom: 10 }}>
            Already within {target_grade} range
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {already_acceptable.map(a => (
              <div key={a.feature} style={{
                fontSize: 11, padding: '4px 10px',
                background: 'var(--positive-lt)', color: 'var(--positive)',
                borderRadius: 20, display: 'flex', alignItems: 'center', gap: 4,
              }}>
                ✓ {a.display_name}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RecCard({ rec, rank }) {
  const dir = rec.direction;
  const color = DIRECTION_COLOR[dir];
  const icon = DIRECTION_ICON[dir];
  const pct = Math.min(100, ((rec.current_value - rec.target_iqr[0]) /
    (rec.target_iqr[1] - rec.target_iqr[0] || 1)) * 100);

  return (
    <div style={{
      background: 'white', borderRadius: 10, padding: '14px 16px',
      boxShadow: 'var(--shadow)', border: '1px solid var(--border)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
        <div>
          <span style={{ fontSize: 10, fontWeight: 700, color, letterSpacing: '0.1em',
            textTransform: 'uppercase', marginRight: 6 }}>
            {icon} {dir}
          </span>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: 15, color: 'var(--burgundy)', fontWeight: 700 }}>
            {rec.display_name}
          </span>
          {rec.unit && <span style={{ fontSize: 11, color: 'var(--slate-lt)', marginLeft: 4 }}>({rec.unit})</span>}
        </div>
        <span style={{ fontSize: 11, color: 'var(--slate-lt)', background: 'var(--parchment)',
          padding: '2px 8px', borderRadius: 12 }}>
          SHAP {rec.shap_value > 0 ? '+' : ''}{rec.shap_value.toFixed(3)}
        </span>
      </div>

      {/* Value range indicator */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11,
          color: 'var(--slate-lt)', marginBottom: 4 }}>
          <span>Current: <strong style={{ color: 'var(--slate)' }}>{rec.current_value}</strong></span>
          <span>Target median: <strong style={{ color }}>{rec.target_value}</strong></span>
        </div>
        {/* IQR range bar */}
        <div style={{ position: 'relative', height: 6, background: 'var(--parchment-dk)', borderRadius: 3 }}>
          <div style={{
            position: 'absolute', left: '15%', right: '15%', top: 0, bottom: 0,
            background: `${color}30`, borderRadius: 3,
          }} title={`Target IQR: ${rec.target_iqr[0]}–${rec.target_iqr[1]}`} />
          {/* Current value marker */}
          <div style={{
            position: 'absolute', top: -3, width: 12, height: 12, borderRadius: '50%',
            background: 'white', border: `2.5px solid ${color}`,
            left: `calc(${Math.max(5, Math.min(95, pct))}% - 6px)`,
            boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
            transition: 'left 0.4s ease',
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10,
          color: 'var(--slate-lt)', marginTop: 3 }}>
          <span>IQR low: {rec.target_iqr[0]}</span>
          <span>IQR high: {rec.target_iqr[1]}</span>
        </div>
      </div>

      <div style={{ fontSize: 12, color: 'var(--slate)', lineHeight: 1.5, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
        {rec.suggestion}
      </div>
    </div>
  );
}

function GradeBadge({ grade, dimmed }) {
  const colors = { Poor: '#C0392B', Average: '#E67E22', Good: '#2D6A4F', Premium: '#B8860B' };
  return (
    <span style={{
      background: dimmed ? 'var(--parchment-dk)' : colors[grade],
      color: dimmed ? 'var(--slate)' : 'white',
      fontWeight: 700, fontSize: 12, padding: '3px 10px', borderRadius: 12,
      fontFamily: 'var(--font-display)',
      border: dimmed ? `2px dashed ${colors[grade]}` : 'none',
    }}>{grade}</span>
  );
}

function Placeholder() {
  return (
    <div style={{ padding: '48px 0', textAlign: 'center', color: 'var(--slate-lt)' }}>
      <div style={{ fontSize: 32, marginBottom: 12 }}>💡</div>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 16, marginBottom: 6 }}>No recommendations yet</div>
      <div style={{ fontSize: 13 }}>Analyse a wine to see how its quality could be improved.</div>
    </div>
  );
}

const headingStyle = {
  fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700,
  color: 'var(--burgundy)', marginBottom: 6,
};