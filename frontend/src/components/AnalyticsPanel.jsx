import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getGlobalImportance, getModelMetrics } from '../api';

const BURGUNDY_SCALE = [
  '#3D0C11','#521118','#6B1A24','#862332','#9E2D40',
  '#B5384F','#C94E65','#D96B7C','#E68FA0','#F0B5C3',
  '#F7D9E1','#FAF0F3',
];

export default function AnalyticsPanel() {
  const [importance, setImportance] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getGlobalImportance(), getModelMetrics()])
      .then(([imp, met]) => { setImportance(imp); setMetrics(met); })
      .catch(e => setError('Could not load analytics. Is the API running?'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  const chartData = importance.feature_importances.map((f, i) => ({
    name: f.display_name.replace(' Dioxide', ' SO₂').replace('Acidity', 'Acid.'),
    value: f.mean_abs_shap,
    fill: BURGUNDY_SCALE[i % BURGUNDY_SCALE.length],
  }));

  const models = metrics.all_results;
  const selected = metrics.selected_model;

  return (
    <div style={{ padding: '8px 0' }}>
      {/* Feature Importance */}
      <Section title="Global Feature Importance" subtitle="Mean |SHAP| value over background dataset — how much each feature moves the quality score on average.">
        <div style={{ height: 300, marginTop: 16 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--slate-lt)' }}
                axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={130}
                tick={{ fontSize: 11, fill: 'var(--slate)' }}
                axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(v) => [v.toFixed(4), 'Mean |SHAP|']}
                contentStyle={{ fontSize: 12, border: '1px solid var(--border)', borderRadius: 6 }}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* Model comparison table */}
      <Section title="Model Comparison" subtitle={`Four regressors trained and compared. Selected: ${selected} (highest macro F1 on grade prediction).`}>
        <div style={{ overflowX: 'auto', marginTop: 16 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border)' }}>
                {['Model', 'MAE', 'RMSE', 'R²', 'Grade Acc.', 'Macro F1', 'Time (s)'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontSize: 11,
                    letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--slate-lt)', fontWeight: 600 }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(models).map(([name, r]) => {
                const isSelected = name === selected;
                return (
                  <tr key={name} style={{
                    background: isSelected ? 'var(--blush)' : 'transparent',
                    borderBottom: '1px solid var(--border)',
                  }}>
                    <td style={{ padding: '9px 10px', fontWeight: isSelected ? 700 : 400,
                      color: isSelected ? 'var(--burgundy)' : 'var(--slate)' }}>
                      {name.replace('_', ' ')}
                      {isSelected && <span style={{ marginLeft: 6, fontSize: 10,
                        background: 'var(--burgundy)', color: 'white',
                        padding: '1px 6px', borderRadius: 10 }}>selected</span>}
                    </td>
                    <MetricCell v={r.regression_metrics.mae} />
                    <MetricCell v={r.regression_metrics.rmse} />
                    <MetricCell v={r.regression_metrics.r2} />
                    <MetricCell v={r.grading_metrics.accuracy} pct />
                    <MetricCell v={r.grading_metrics.macro_f1} highlight={isSelected} />
                    <MetricCell v={r.train_time_sec} />
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Per-grade breakdown for selected model */}
        <div style={{ marginTop: 24 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--slate)', marginBottom: 12,
            textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Per-grade metrics — {selected}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            {Object.entries(models[selected].grading_metrics.per_grade).map(([grade, m]) => (
              <GradeCard key={grade} grade={grade} metrics={m} />
            ))}
          </div>
        </div>
      </Section>
    </div>
  );
}

function MetricCell({ v, pct, highlight }) {
  const val = pct ? `${(v * 100).toFixed(1)}%` : v?.toFixed(4);
  return (
    <td style={{ padding: '9px 10px', color: highlight ? 'var(--burgundy)' : 'var(--slate)',
      fontWeight: highlight ? 700 : 400 }}>
      {val}
    </td>
  );
}

const GRADE_COLORS = { Poor: '#C0392B', Average: '#E67E22', Good: '#2D6A4F', Premium: '#B8860B' };

function GradeCard({ grade, metrics }) {
  const color = GRADE_COLORS[grade];
  return (
    <div style={{ background: 'white', borderRadius: 8, padding: '12px', border: '1px solid var(--border)',
      boxShadow: 'var(--shadow)' }}>
      <div style={{ fontSize: 12, fontWeight: 700, color, marginBottom: 8, fontFamily: 'var(--font-display)' }}>
        {grade}
      </div>
      {[['Precision', metrics.precision], ['Recall', metrics.recall], ['F1', metrics.f1]].map(([k, v]) => (
        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
          <span style={{ color: 'var(--slate-lt)' }}>{k}</span>
          <span style={{ fontWeight: 600, color: 'var(--slate)' }}>{(v * 100).toFixed(1)}%</span>
        </div>
      ))}
      <div style={{ fontSize: 10, color: 'var(--slate-lt)', marginTop: 6, borderTop: '1px solid var(--border)', paddingTop: 4 }}>
        n = {metrics.support}
      </div>
    </div>
  );
}

function Section({ title, subtitle, children }) {
  return (
    <div style={{ marginBottom: 36 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700,
        color: 'var(--burgundy)', marginBottom: 4 }}>{title}</div>
      {subtitle && <div style={{ fontSize: 12, color: 'var(--slate-lt)', lineHeight: 1.6 }}>{subtitle}</div>}
      {children}
    </div>
  );
}

function LoadingState() {
  return (
    <div style={{ padding: '60px 0', textAlign: 'center', color: 'var(--slate-lt)' }}>
      <div style={{ fontSize: 24, marginBottom: 12 }}>📊</div>
      <div style={{ fontSize: 14 }}>Loading analytics…</div>
    </div>
  );
}

function ErrorState({ message }) {
  return (
    <div style={{ padding: '32px', background: 'var(--negative-lt)', borderRadius: 10,
      border: '1px solid var(--negative)', color: 'var(--negative)', fontSize: 13 }}>
      ⚠️ {message}
    </div>
  );
}