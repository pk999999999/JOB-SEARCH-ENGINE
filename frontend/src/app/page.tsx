'use client';
import { useEffect, useState } from 'react';
import { getStats, type Stats } from '@/lib/api';
import { Users, Calendar, Briefcase, Zap, MapPin, Code2, TrendingUp } from 'lucide-react';

const STAT_COLORS = ['var(--cyan)', 'var(--lime)', 'var(--pink)', 'var(--amber)'];

export default function DashboardPage() {
  const [stats,   setStats]   = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');

  useEffect(() => {
    getStats().then(setStats).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="loading-container" style={{ minHeight: '60vh' }}>
      <div className="loading-spinner" /><p>Loading dashboard…</p>
    </div>
  );

  if (error) return (
    <div className="empty-state" style={{ minHeight: '60vh' }}>
      <div className="icon" style={{ color: 'var(--warn)' }}><Zap size={44} /></div>
      <h3>Cannot reach API</h3>
      <p style={{ maxWidth: 380, margin: '8px auto 16px', color: 'var(--tx-2)', lineHeight: 1.6 }}>
        Start the backend on port 8000.
      </p>
      <code style={{
        display: 'block', background: 'var(--bg-2)', border: '1px solid var(--b2)',
        padding: '10px 16px', borderRadius: 8, fontFamily: 'var(--font-mono)',
        fontSize: '0.82rem', maxWidth: 320, margin: '0 auto 12px', color: 'var(--cyan)',
      }}>
        uvicorn api.main:app --reload
      </code>
      <p style={{ color: 'var(--err)', fontSize: '0.75rem' }}>{error}</p>
    </div>
  );

  const STAT_ITEMS = [
    { Icon: Users,    value: stats?.total_candidates.toLocaleString() ?? '0', label: 'Total candidates',     id:'stat-candidates' },
    { Icon: Calendar, value: stats?.avg_experience_years.toFixed(1) ?? '0',  label: 'Avg experience (yrs)', id:'stat-experience' },
    { Icon: Briefcase,value: String(stats?.total_jobs ?? '0'),                label: 'Active job specs',     id:'stat-jobs' },
    { Icon: Zap,      value: 'AI',                                             label: 'Ranker model active',  id:'stat-engine' },
  ];

  const maxLoc = Math.max(...(stats?.top_locations.map(l => l.count) ?? [1]));

  return (
    <>
      <div className="page-header">
        <h1>
          <TrendingUp size={26} color="var(--cyan)" aria-hidden="true" />
          <span className="title-text">Dashboard</span>
        </h1>
        <p>Live analytics and candidate database stats</p>
      </div>

      {/* ── Stat cards ── */}
      <div className="stats-grid">
        {STAT_ITEMS.map(({ Icon, value, label, id }, i) => (
          <div className="card stat-card" key={id} id={id}>
            <div className="stat-icon">
              <Icon size={20} color={STAT_COLORS[i]} aria-hidden="true" />
            </div>
            <div className="stat-value" style={{ color: STAT_COLORS[i], WebkitTextFillColor: 'unset', background: 'none' }}>
              {value}
            </div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      {/* ── Charts ── */}
      <div className="two-col" style={{ gridTemplateColumns: '1.3fr 1fr' }}>

        {/* Top locations */}
        <div className="card section">
          <h3 className="section-title">
            <MapPin size={13} color="var(--cyan)" aria-hidden="true" />
            Top locations
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {stats?.top_locations.map((loc, i) => {
              const pct = Math.round((loc.count / maxLoc) * 100);
              const color = [
                'var(--cyan)', 'var(--lime)', 'var(--pink)',
                'var(--amber)', 'var(--s-edu)',
              ][i] ?? 'var(--cyan)';
              return (
                <div
                  key={i}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '10px 12px',
                    background: 'var(--bg-2)',
                    border: '1px solid var(--b1)',
                    borderRadius: 10,
                    transition: 'border-color 0.12s',
                    cursor: 'default',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = `${color}44`)}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--b1)')}
                >
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.68rem',
                    fontWeight: 700,
                    color: color,
                    width: 18,
                    flexShrink: 0,
                  }}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span style={{ flex: 1, fontWeight: 600, fontSize: '0.875rem', color: 'var(--tx-1)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {loc.location}
                  </span>
                  <div style={{ width: 100, flexShrink: 0 }}>
                    <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 99, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', width: `${pct}%`,
                        background: color,
                        borderRadius: 99,
                        boxShadow: `0 0 6px ${color}`,
                        transition: 'width 0.9s cubic-bezier(.34,1.56,.64,1)',
                      }} />
                    </div>
                  </div>
                  <span style={{ fontSize: '0.78rem', color, fontWeight: 800, minWidth: 32, textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {loc.count}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Skills cloud */}
        <div className="card section">
          <h3 className="section-title">
            <Code2 size={13} color="var(--lime)" aria-hidden="true" />
            Core skills
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignContent: 'flex-start' }}>
            {stats?.skill_distribution.map((skill, i) => {
              const max = stats.skill_distribution[0]?.count ?? 1;
              const ratio = skill.count / max;
              const fs = 10 + Math.round(ratio * 6);
              const colors = ['var(--cyan)', 'var(--lime)', 'var(--pink)', 'var(--amber)', 'var(--s-edu)'];
              const c = colors[i % colors.length];
              return (
                <div
                  key={i}
                  title={`${skill.skill}: ${skill.count}`}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    padding: '5px 11px',
                    background: 'var(--bg-2)',
                    border: `1px solid ${c}22`,
                    borderRadius: 99,
                    cursor: 'default',
                    transition: 'all 0.12s',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = `${c}55`;
                    e.currentTarget.style.background = `${c}0d`;
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = `${c}22`;
                    e.currentTarget.style.background = 'var(--bg-2)';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  <span style={{ fontSize: `${fs}px`, fontWeight: 700, color: 'var(--tx-1)' }}>
                    {skill.skill}
                  </span>
                  <span style={{
                    fontSize: '0.58rem', fontWeight: 700,
                    padding: '1px 5px',
                    background: `${c}18`,
                    borderRadius: 4,
                    color: c,
                  }}>
                    {skill.count}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}