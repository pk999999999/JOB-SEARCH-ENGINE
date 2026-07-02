'use client';
import { useState } from 'react';
import { rankCandidates, type RankResponse } from '@/lib/api';
import CandidateCard from '@/components/CandidateCard';
import { Search, AlertCircle, BarChart3, Trophy, Timer, Target, Sparkles } from 'lucide-react';

const STAT_COLORS = ['var(--cyan)', 'var(--lime)', 'var(--pink)', 'var(--amber)'];

export default function SearchPage() {
  const [jobTitle,        setJobTitle]        = useState('Senior AI Engineer');
  const [description,     setDescription]     = useState('We are looking for a Senior AI Engineer to design, build, and deploy production-grade machine learning systems.');
  const [requiredSkills,  setRequiredSkills]  = useState('python, machine learning, deep learning, natural language processing, pytorch, transformers');
  const [preferredSkills, setPreferredSkills] = useState('retrieval systems, ranking algorithms, llm, vector databases, docker, kubernetes, aws');
  const [topK,    setTopK]    = useState(20);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<RankResponse | null>(null);
  const [error,   setError]   = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true); setError(''); setResults(null);
    try {
      const r = await rankCandidates({
        job_title: jobTitle, description,
        required_skills:  requiredSkills.split(',').map(s => s.trim()).filter(Boolean),
        preferred_skills: preferredSkills.split(',').map(s => s.trim()).filter(Boolean),
        top_k: topK,
      });
      setResults(r);
    } catch (e: any) { setError(e.message || 'Ranking failed'); }
    finally { setLoading(false); }
  };

  const reqArr  = requiredSkills.split(',').map(s => s.trim()).filter(Boolean);
  const prefArr = preferredSkills.split(',').map(s => s.trim()).filter(Boolean);

  return (
    <>
      <div className="page-header">
        <h1>
          <Search size={24} color="var(--cyan)" aria-hidden="true" />
          <span className="title-text">Search & Rank</span>
        </h1>
        <p>Describe the role — the AI finds who genuinely fits, not who keyword-matches</p>
      </div>

      {/* ── Form ── */}
      <div className="card" style={{ marginBottom: 32 }}>

        {/* Card header strip */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          marginBottom: 20, paddingBottom: 16,
          borderBottom: '1px solid var(--b1)',
        }}>
          <Sparkles size={15} color="var(--cyan)" aria-hidden="true" />
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--tx-2)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Job specification
          </span>
        </div>

        <form className="search-form" onSubmit={handleSubmit} id="rank-form">
          <div className="search-form-row">
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" htmlFor="job-title">Job title</label>
              <input id="job-title" className="form-input" value={jobTitle}
                onChange={e => setJobTitle(e.target.value)} placeholder="Senior AI Engineer" />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" htmlFor="top-k">Top K results</label>
              <input id="top-k" className="form-input" type="number" min={1} max={100}
                value={topK} onChange={e => setTopK(Number(e.target.value))} />
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label" htmlFor="description">Job description</label>
            <textarea id="description" className="form-textarea" rows={3}
              value={description} onChange={e => setDescription(e.target.value)}
              placeholder="Describe the ideal candidate and what success looks like…" />
          </div>

          <div className="search-form-row">
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" htmlFor="required-skills">
                Required skills
                <span style={{ marginLeft: 6, color: 'var(--err)', fontSize: '0.6rem' }}>*</span>
              </label>
              <textarea id="required-skills" className="form-textarea" rows={2}
                value={requiredSkills} onChange={e => setRequiredSkills(e.target.value)}
                placeholder="python, pytorch, machine learning…" />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" htmlFor="preferred-skills">Preferred skills</label>
              <textarea id="preferred-skills" className="form-textarea" rows={2}
                value={preferredSkills} onChange={e => setPreferredSkills(e.target.value)}
                placeholder="docker, kubernetes, aws…" />
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-full"
            disabled={loading}
            id="rank-submit"
            style={{
              padding: '13px',
              background: loading ? 'var(--bg-3)' : 'rgba(0,229,255,0.08)',
              border: '1px solid',
              borderColor: loading ? 'var(--b1)' : 'rgba(0,229,255,0.4)',
              color: loading ? 'var(--tx-3)' : 'var(--cyan)',
              fontWeight: 800,
              fontSize: '0.9rem',
              letterSpacing: '0.02em',
              boxShadow: loading ? 'none' : '0 0 28px rgba(0,229,255,0.1)',
              gap: 8,
            }}
          >
            {loading ? (
              <><span className="loading-spinner" style={{ width:16, height:16, borderWidth:2, margin:0 }} /> Ranking candidates…</>
            ) : (
              <><Target size={17} aria-hidden="true" /> Run AI ranking</>
            )}
          </button>
        </form>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="error-banner" role="alert">
          <AlertCircle size={15} aria-hidden="true" /> {error}
        </div>
      )}

      {/* ── Results ── */}
      {results && (
        <>
          {/* Stats row */}
          <div className="stats-grid" style={{ marginBottom: 24 }}>
            {[
              { Icon: BarChart3, value: results.total_candidates.toLocaleString(), label: 'Candidates scored' },
              { Icon: Trophy,    value: String(results.ranked_count),              label: 'Top results' },
              { Icon: Timer,     value: `${results.elapsed_seconds}s`,             label: 'Ranking time' },
              { Icon: Target,    value: results.candidates[0] ? (results.candidates[0].score * 100).toFixed(1) : '—', label: 'Top score' },
            ].map(({ Icon, value, label }, i) => (
              <div key={i} className="card stat-card">
                <div className="stat-icon"><Icon size={18} color={STAT_COLORS[i]} /></div>
                <div className="stat-value" style={{ color: STAT_COLORS[i], background: 'none', WebkitTextFillColor: 'unset' }}>
                  {value}
                </div>
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>

          {/* Candidate cards */}
          <div className="section">
            <h2 className="section-title">
              Results for &ldquo;{results.job_title}&rdquo;
              <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--tx-3)', textTransform: 'none', letterSpacing: 0, fontWeight: 400 }}>
                {results.ranked_count} candidates
              </span>
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {results.candidates.map((c, i) => (
                <CandidateCard key={c.candidate_id} candidate={c} index={i}
                  searchRequiredSkills={reqArr} searchPreferredSkills={prefArr} />
              ))}
            </div>
          </div>
        </>
      )}

      {/* ── Empty ── */}
      {!results && !loading && !error && (
        <div className="empty-state">
          <div className="icon" style={{ color: 'var(--tx-3)' }}><Search size={48} /></div>
          <h3>Ready to rank</h3>
          <p>Fill in the job spec above and run the AI ranking engine.</p>
        </div>
      )}
    </>
  );
}