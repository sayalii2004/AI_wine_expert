import React from 'react';

export default function ExplanationPanel({ data }) {
  if (!data) return <Placeholder />;

  const { predicted_quality, expected_quality, shap_values, predicted_grade } = data;
  if (!shap_values?.length) return <Placeholder />;

  const maxAbs = Math.max(...shap_values.map(s => Math.abs(s.shap_value)), 0.01);

  return (
    <div style={{ padding: '8px 0' }}>
      <div style={headingStyle}>SHAP Feature Contributions</div>
      <p style={subStyle}>
        Baseline quality: <strong>{expected_quality}</strong> → Predicted: <strong>{predicted_quality}</strong>.
        Each bar shows how much a feature pushed the score up or down.
      </p>

      {/* Waterfall bars */}
      <div style={{ marginTop: 20 }}>
        {shap_values.map((entry, i) => {
          const isPos = entry.shap_value >= 0;
          const pct = (Math.abs(entry.shap_value) / maxAbs) * 100;
          const color = isPos ? 'var(--positive)' : 'var(--negative)';
          const bg = isPos ? 'var(--positive-lt)' : 'var(--negative-lt)';
          return (
            <div key={entry.feature} style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: 'var(--slate)', fontWeight: 500 }}>
                  {entry.display_name}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--slate-lt)' }}>
                    val: {entry.value}
                  </span>
                  <span style={{
                    fontSize: 12, fontWeight: 700, color,
                    minWidth: 50, textAlign: 'right',
                  }}>
                    {isPos ? '+' : ''}{entry.shap_value.toFixed(3)}
                  </span>
                </div>
              </div>
              <div style={{ background: 'var(--parchment-dk)', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                <div style={{
                  width: `${pct}%`, height: '100%', background: color,
                  marginLeft: isPos ? 0 : 'auto',
                  borderRadius: 4,
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 20, marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
        <LegendItem color="var(--positive)" label="Raised quality" />
        <LegendItem color="var(--negative)" label="Lowered quality" />
      </div>
    </div>
  );
}

function LegendItem({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--slate-lt)' }}>
      <div style={{ width: 12, height: 12, borderRadius: 2, background: color }} />
      {label}
    </div>
  );
}

function Placeholder() {
  return (
    <div style={{ padding: '48px 0', textAlign: 'center', color: 'var(--slate-lt)' }}>
      <div style={{ fontSize: 32, marginBottom: 12 }}>🔬</div>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 16, marginBottom: 6 }}>No explanation yet</div>
      <div style={{ fontSize: 13 }}>Analyse a wine to see which features drove the prediction.</div>
    </div>
  );
}

const headingStyle = {
  fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700,
  color: 'var(--burgundy)', marginBottom: 6,
};
const subStyle = { fontSize: 13, color: 'var(--slate-lt)', lineHeight: 1.6 };