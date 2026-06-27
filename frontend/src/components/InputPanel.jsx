import React from 'react';

const FEATURES = [
  { key: 'fixed acidity',        label: 'Fixed Acidity',         unit: 'g/L',  min: 3.8,  max: 15.9, step: 0.1,  default: 7.5 },
  { key: 'volatile acidity',     label: 'Volatile Acidity',      unit: 'g/L',  min: 0.08, max: 1.58, step: 0.01, default: 0.4 },
  { key: 'citric acid',          label: 'Citric Acid',           unit: 'g/L',  min: 0.0,  max: 1.66, step: 0.01, default: 0.3 },
  { key: 'residual sugar',       label: 'Residual Sugar',        unit: 'g/L',  min: 0.6,  max: 65.8, step: 0.1,  default: 3.0 },
  { key: 'chlorides',            label: 'Chlorides',             unit: 'g/L',  min: 0.009,max: 0.611,step: 0.001,default: 0.05},
  { key: 'free sulfur dioxide',  label: 'Free SO₂',              unit: 'mg/L', min: 1.0,  max: 289,  step: 1,    default: 30  },
  { key: 'total sulfur dioxide', label: 'Total SO₂',             unit: 'mg/L', min: 6.0,  max: 440,  step: 1,    default: 100 },
  { key: 'density',              label: 'Density',               unit: 'g/cm³',min: 0.987,max: 1.039,step: 0.0001,default:0.994},
  { key: 'pH',                   label: 'pH',                    unit: '',     min: 2.72, max: 4.01, step: 0.01, default: 3.2 },
  { key: 'sulphates',            label: 'Sulphates',             unit: 'g/L',  min: 0.22, max: 2.0,  step: 0.01, default: 0.5 },
  { key: 'alcohol',              label: 'Alcohol',               unit: '% vol',min: 8.0,  max: 14.9, step: 0.1,  default: 10.5},
];

export const DEFAULT_FEATURES = Object.fromEntries(
  FEATURES.map(f => [f.key, f.default])
);

const styles = {
  panel: {
    width: 280, flexShrink: 0, background: 'var(--burgundy)', color: 'white',
    display: 'flex', flexDirection: 'column', height: '100vh',
    position: 'sticky', top: 0, overflowY: 'auto',
  },
  header: {
    padding: '28px 24px 16px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
  },
  title: {
    fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 700,
    color: 'white', lineHeight: 1.2, marginBottom: 4,
  },
  subtitle: {
    fontSize: 11, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  typeSelector: {
    padding: '14px 24px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
  },
  typeLabel: {
    fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase',
    color: 'rgba(255,255,255,0.5)', marginBottom: 8,
  },
  typeButtons: { display: 'flex', gap: 8 },
  typeBtn: (active) => ({
    flex: 1, padding: '7px 0', border: '1px solid',
    borderColor: active ? 'var(--gold)' : 'rgba(255,255,255,0.2)',
    background: active ? 'var(--gold)' : 'transparent',
    color: active ? 'var(--burgundy)' : 'rgba(255,255,255,0.8)',
    borderRadius: 6, fontSize: 13, fontWeight: active ? 600 : 400,
    transition: 'all 0.2s',
    cursor: 'pointer',
  }),
  featureList: { padding: '12px 24px', flex: 1 },
  featureRow: { marginBottom: 14 },
  featureHeader: { display: 'flex', justifyContent: 'space-between', marginBottom: 3 },
  featureLabel: { fontSize: 11, color: 'rgba(255,255,255,0.7)', letterSpacing: '0.03em' },
  featureValue: { fontSize: 12, fontWeight: 600, color: 'var(--gold-light)' },
  slider: { width: '100%', height: 3, margin: '4px 0' },
  analyseBtn: {
    margin: '4px 24px 24px', padding: '13px', background: 'var(--gold)',
    border: 'none', borderRadius: 8, color: 'var(--burgundy)',
    fontFamily: 'var(--font-display)', fontSize: 15, fontWeight: 700,
    letterSpacing: '0.03em', cursor: 'pointer',
    boxShadow: '0 2px 12px rgba(184,134,11,0.4)',
    transition: 'all 0.2s',
  },
  resetBtn: {
    margin: '0 24px 16px', padding: '9px', background: 'transparent',
    border: '1px solid rgba(255,255,255,0.2)', borderRadius: 8,
    color: 'rgba(255,255,255,0.6)', fontSize: 12, cursor: 'pointer',
    transition: 'all 0.2s',
  },
};

export default function InputPanel({ features, wineType, onChange, onWineTypeChange, onAnalyse, loading }) {
  return (
    <aside style={styles.panel}>
      <div style={styles.header}>
        <div style={styles.title}>AI Wine Expert</div>
        <div style={styles.subtitle}>Physicochemical Analysis</div>
      </div>

      <div style={styles.typeSelector}>
        <div style={styles.typeLabel}>Wine Type</div>
        <div style={styles.typeButtons}>
          {['red', 'white'].map(t => (
            <button key={t} style={styles.typeBtn(wineType === t)}
              onClick={() => onWineTypeChange(t)}>
              {t === 'red' ? '🍷 Red' : '🥂 White'}
            </button>
          ))}
        </div>
      </div>

      <div style={styles.featureList}>
        {FEATURES.map(f => (
          <div key={f.key} style={styles.featureRow}>
            <div style={styles.featureHeader}>
              <span style={styles.featureLabel}>{f.label}</span>
              <span style={styles.featureValue}>
                {features[f.key]?.toFixed(f.step < 0.01 ? 4 : f.step < 0.1 ? 3 : f.step < 1 ? 2 : 1)}
                {f.unit ? ` ${f.unit}` : ''}
              </span>
            </div>
            <input
              type="range" min={f.min} max={f.max} step={f.step}
              value={features[f.key] ?? f.default}
              onChange={e => onChange(f.key, parseFloat(e.target.value))}
              style={styles.slider}
            />
          </div>
        ))}
      </div>

      <button style={styles.analyseBtn} onClick={onAnalyse} disabled={loading}>
        {loading ? 'Analysing…' : 'Analyse Wine'}
      </button>
      <button style={styles.resetBtn} onClick={() => {
        FEATURES.forEach(f => onChange(f.key, f.default));
      }}>
        Reset to defaults
      </button>
    </aside>
  );
}