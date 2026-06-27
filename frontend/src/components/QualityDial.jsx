import React from 'react';

const GRADE_COLOR = {
  Poor:    '#C0392B',
  Average: '#E67E22',
  Good:    '#2D6A4F',
  Premium: '#B8860B',
};

const GRADE_LABEL = {
  Poor:    'Poor',
  Average: 'Average',
  Good:    'Good',
  Premium: 'Premium',
};

// Score 0–10 maps to arc from -210° to +30° (240° sweep)
function scoreToAngle(score) {
  const clamped = Math.max(0, Math.min(10, score));
  return -210 + (clamped / 10) * 240;
}

function polarToXY(cx, cy, r, angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx, cy, r, startAngle, endAngle) {
  const s = polarToXY(cx, cy, r, startAngle);
  const e = polarToXY(cx, cy, r, endAngle);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

export default function QualityDial({ score, grade, loading }) {
  const cx = 110, cy = 110, r = 80;
  const needleAngle = score != null ? scoreToAngle(score) : -210;
  const needleTip = polarToXY(cx, cy, r - 8, needleAngle);
  const needleBack = polarToXY(cx, cy, 14, needleAngle + 180);
  const color = grade ? GRADE_COLOR[grade] : '#DDD4C0';

  // Grade arc segments
  const segments = [
    { grade: 'Poor',    start: -210, end: -138, color: '#FAD7D3' },
    { grade: 'Average', start: -138, end:  -54, color: '#FEF0D9' },
    { grade: 'Good',    start:  -54, end:   18, color: '#D4EDE4' },
    { grade: 'Premium', start:   18, end:   30, color: '#FFF3CD' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg width={220} height={150} viewBox="0 0 220 150" aria-label="Quality score dial">
        {/* Track */}
        <path
          d={arcPath(cx, cy, r, -210, 30)}
          fill="none" stroke="var(--parchment-dk)" strokeWidth={18} strokeLinecap="round"
        />

        {/* Grade segments */}
        {segments.map(seg => (
          <path
            key={seg.grade}
            d={arcPath(cx, cy, r, seg.start, seg.end)}
            fill="none" stroke={seg.color} strokeWidth={18}
          />
        ))}

        {/* Fill to current score */}
        {score != null && (
          <path
            d={arcPath(cx, cy, r, -210, needleAngle)}
            fill="none" stroke={color} strokeWidth={18} strokeLinecap="round"
            style={{ transition: 'all 0.7s cubic-bezier(.4,0,.2,1)' }}
          />
        )}

        {/* Tick marks at 0, 2, 4, 6, 8, 10 */}
        {[0, 2, 4, 6, 8, 10].map(v => {
          const a = scoreToAngle(v);
          const inner = polarToXY(cx, cy, r - 13, a);
          const outer = polarToXY(cx, cy, r + 5, a);
          const label = polarToXY(cx, cy, r + 16, a);
          return (
            <g key={v}>
              <line x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y}
                stroke="var(--slate-lt)" strokeWidth={1.5} />
              <text x={label.x} y={label.y} textAnchor="middle" dominantBaseline="middle"
                fontSize={9} fill="var(--slate-lt)" fontFamily="var(--font-body)">{v}</text>
            </g>
          );
        })}

        {/* Needle */}
        {score != null && (
          <g style={{ transition: 'all 0.7s cubic-bezier(.4,0,.2,1)' }}>
            <line
              x1={needleBack.x} y1={needleBack.y}
              x2={needleTip.x}  y2={needleTip.y}
              stroke={color} strokeWidth={3} strokeLinecap="round"
            />
            <circle cx={cx} cy={cy} r={7} fill={color} />
            <circle cx={cx} cy={cy} r={3} fill="white" />
          </g>
        )}

        {/* Centre score */}
        {loading ? (
          <text x={cx} y={cy + 32} textAnchor="middle" fontSize={13}
            fill="var(--slate-lt)" fontFamily="var(--font-body)">analysing…</text>
        ) : score != null ? (
          <>
            <text x={cx} y={cy + 28} textAnchor="middle" fontSize={28}
              fontWeight="700" fill={color} fontFamily="var(--font-display)">
              {score.toFixed(1)}
            </text>
            <text x={cx} y={cy + 44} textAnchor="middle" fontSize={11}
              fill="var(--slate-lt)" fontFamily="var(--font-body)">out of 10</text>
          </>
        ) : (
          <text x={cx} y={cy + 34} textAnchor="middle" fontSize={12}
            fill="var(--slate-lt)" fontFamily="var(--font-body)">Awaiting input</text>
        )}
      </svg>

      {/* Grade badge */}
      {grade && (
        <div style={{
          background: color, color: 'white', fontFamily: 'var(--font-display)',
          fontSize: 15, fontWeight: 700, letterSpacing: '0.06em',
          padding: '4px 20px', borderRadius: 20,
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          transition: 'all 0.4s ease',
        }}>
          {GRADE_LABEL[grade]}
        </div>
      )}
    </div>
  );
}