'use client';
import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Download, MapPin, Calendar, DollarSign, Briefcase, GraduationCap, Wrench, BarChart2, ShieldCheck } from 'lucide-react';
import { getCandidate, type CandidateDetail } from '@/lib/api';
import ScoreBar from '@/components/ScoreBar';

function Content() {
  const id = useSearchParams().get('id') || '';
  const [candidate, setCandidate] = useState<CandidateDetail | null>(null);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState('');

  useEffect(() => {
    if (!id) return;
    getCandidate(id).then(setCandidate).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="loading-container" style={{ minHeight:'50vh' }}><div className="loading-spinner" /><p>Loading profile…</p></div>;

  if (error || !candidate) return (
    <div className="empty-state">
      <div className="icon" style={{ color:'var(--err)' }}><ShieldCheck size={44} /></div>
      <h3>Not found</h3>
      <p style={{ marginBottom:16 }}>{error}</p>
      <Link href="/candidates" className="btn btn-secondary"><ArrowLeft size={13} /> Back</Link>
    </div>
  );

  const bs = candidate.behavioral_signals;
  const API = process.env.NEXT_PUBLIC_API_URL !== undefined && process.env.NEXT_PUBLIC_API_URL !== ''
    ? process.env.NEXT_PUBLIC_API_URL
    : (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');

  return (
    <>
      {/* ── Page header ── */}
      <div className="page-header" style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div>
          <Link href="/candidates" style={{
            display:'inline-flex', alignItems:'center', gap:4,
            fontSize:'0.75rem', color:'var(--tx-3)',
            marginBottom:10, transition:'color 0.12s',
          }}
          onMouseEnter={e=>(e.currentTarget.style.color='var(--cyan)')}
          onMouseLeave={e=>(e.currentTarget.style.color='var(--tx-3)')}
          >
            <ArrowLeft size={12} /> Back to candidates
          </Link>
          <h1 style={{ fontSize:'1.6rem' }}>
            <span className="title-text">
              {candidate.current_title || candidate.headline || candidate.candidate_id}
            </span>
          </h1>
          {candidate.headline && candidate.current_title && (
            <p style={{ color:'var(--tx-2)', marginTop:4 }}>{candidate.headline}</p>
          )}
        </div>
        <a href={`${API}/api/candidates/${candidate.candidate_id}/resume`}
          className="btn btn-primary" target="_blank" rel="noopener noreferrer" style={{ flexShrink:0 }}>
          <Download size={14} /> Resume
        </a>
      </div>

      <div className="two-col">

        {/* ── Main column ── */}
        <div>
          {candidate.summary && (
            <div className="card section">
              <h3 className="section-title">Summary</h3>
              <p style={{ color:'var(--tx-2)', lineHeight:1.75, fontSize:'0.9rem' }}>{candidate.summary}</p>
            </div>
          )}

          <div className="card section">
            <h3 className="section-title"><Briefcase size={12} color="var(--cyan)" /> Career history</h3>
            {candidate.career_history.length > 0 ? (
              <div className="timeline">
                {candidate.career_history.map((e, i) => (
                  <div key={i} className="timeline-item">
                    <div className="timeline-dot" />
                    <div className="timeline-content">
                      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', gap:8 }}>
                        <div>
                          <div style={{ fontWeight:700, fontSize:'0.9rem', color:'var(--tx-1)' }}>{e.title}</div>
                          <div style={{ color:'var(--cyan)', fontSize:'0.82rem', fontWeight:600, marginTop:2 }}>{e.company}</div>
                        </div>
                        <div style={{ textAlign:'right', flexShrink:0 }}>
                          {e.is_current
                            ? <span className="tag tag-success" style={{ fontSize:'0.62rem' }}>Current</span>
                            : <span className="tag" style={{ fontSize:'0.62rem', opacity:0.5 }}>Past</span>
                          }
                          <div style={{ fontSize:'0.68rem', color:'var(--tx-3)', marginTop:4, fontFamily:'var(--font-mono)' }}>
                            {e.duration_months}mo
                          </div>
                        </div>
                      </div>
                      {e.description && (
                        <p style={{ color:'var(--tx-2)', fontSize:'0.8rem', marginTop:8, lineHeight:1.6,
                          borderLeft:'1px solid var(--b2)', paddingLeft:10 }}>
                          {e.description}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : <p style={{ color:'var(--tx-3)' }}>No career history.</p>}
          </div>

          <div className="card section">
            <h3 className="section-title"><GraduationCap size={12} color="var(--lime)" /> Education</h3>
            {candidate.education.length > 0
              ? candidate.education.map((e, i) => (
                <div key={i} style={{
                  padding:'12px 0',
                  borderBottom: i < candidate.education.length-1 ? '1px solid var(--b1)' : 'none',
                  display:'flex', justifyContent:'space-between', alignItems:'center', gap:12,
                }}>
                  <div>
                    <span style={{ fontWeight:700, color:'var(--tx-1)' }}>{e.degree}</span>
                    {e.field_of_study && <span style={{ color:'var(--tx-2)' }}> in {e.field_of_study}</span>}
                  </div>
                  <div style={{ textAlign:'right', flexShrink:0 }}>
                    <div style={{ fontWeight:700, color:'var(--cyan)', fontSize:'0.83rem' }}>{e.institution}</div>
                    {e.year > 0 && <div style={{ fontSize:'0.72rem', color:'var(--tx-3)', fontFamily:'var(--font-mono)', marginTop:2 }}>'{String(e.year).slice(-2)}</div>}
                  </div>
                </div>
              ))
              : <p style={{ color:'var(--tx-3)' }}>No education data.</p>
            }
          </div>
        </div>

        {/* ── Sidebar column ── */}
        <div>
          <div className="card section">
            <h3 className="section-title">Quick info</h3>
            <div className="detail-widget-grid">
              <div className="detail-widget">
                <span style={{ fontSize:'0.62rem', color:'var(--tx-3)', fontWeight:700, display:'flex', alignItems:'center', gap:3, textTransform:'uppercase', letterSpacing:'0.06em' }}>
                  <MapPin size={10} /> Location
                </span>
                <span style={{ fontWeight:700, fontSize:'0.875rem', color:'var(--tx-1)', marginTop:3 }}>
                  {candidate.location || 'Unknown'}
                </span>
              </div>
              <div className="detail-widget">
                <span style={{ fontSize:'0.62rem', color:'var(--tx-3)', fontWeight:700, display:'flex', alignItems:'center', gap:3, textTransform:'uppercase', letterSpacing:'0.06em' }}>
                  <Calendar size={10} /> Experience
                </span>
                <span style={{ fontWeight:700, fontSize:'0.875rem', color:'var(--cyan)', marginTop:3, fontFamily:'var(--font-mono)' }}>
                  {candidate.years_of_experience.toFixed(1)}y
                </span>
              </div>
              {candidate.salary_min !== null && (
                <div className="detail-widget" style={{ gridColumn:'span 2' }}>
                  <span style={{ fontSize:'0.62rem', color:'var(--tx-3)', fontWeight:700, display:'flex', alignItems:'center', gap:3, textTransform:'uppercase', letterSpacing:'0.06em' }}>
                    <DollarSign size={10} /> Salary
                  </span>
                  <span style={{ fontWeight:700, fontSize:'0.875rem', color:'var(--lime)', marginTop:3 }}>
                    ₹{(candidate.salary_min/100000).toFixed(1)}L – ₹{((candidate.salary_max||0)/100000).toFixed(1)}L
                  </span>
                </div>
              )}
              <div className="detail-widget" style={{ gridColumn:'span 2' }}>
                <span style={{ fontSize:'0.62rem', color:'var(--tx-3)', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.06em' }}>ID</span>
                <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.65rem', color:'var(--tx-3)', marginTop:3, wordBreak:'break-all' }}>
                  {candidate.candidate_id}
                </span>
              </div>
            </div>
          </div>

          <div className="card section">
            <h3 className="section-title">
              <Wrench size={12} color="var(--amber)" /> Skills
              <span style={{ marginLeft:'auto', fontFamily:'var(--font-mono)', fontSize:'0.65rem', color:'var(--tx-3)', fontWeight:400, textTransform:'none', letterSpacing:0 }}>
                {candidate.skills.length}
              </span>
            </h3>
            <div style={{ display:'flex', flexWrap:'wrap', gap:5 }}>
              {candidate.skills.map((s, i) => (
                <span key={i} className="tag">
                  {s.name}
                  {s.proficiency && <span style={{ marginLeft:4, opacity:0.5, fontSize:'0.62rem' }}>({s.proficiency})</span>}
                </span>
              ))}
            </div>
          </div>

          <div className="card section">
            <h3 className="section-title"><BarChart2 size={12} color="var(--pink)" /> Signals</h3>
            <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
              <ScoreBar label="Recruiter response" value={bs.recruiter_response_rate}  color="var(--s-skills)"  animate />
              <ScoreBar label="Interview completion" value={bs.interview_completion_rate} color="var(--s-exp)"  animate />
              <ScoreBar label="Offer acceptance"    value={bs.offer_acceptance_rate}     color="var(--s-career)" animate />
            </div>

            <div style={{ borderTop:'1px solid var(--b1)', marginTop:14, paddingTop:14 }}>
              <p style={{ fontSize:'0.65rem', fontWeight:700, color:'var(--tx-3)', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8 }}>
                Identity
              </p>
              <div style={{ display:'flex', flexWrap:'wrap', gap:5 }}>
                {[
                  { ok: bs.verified_email,     label: 'Email' },
                  { ok: bs.verified_phone,     label: 'Phone' },
                  { ok: bs.linkedin_connected, label: 'LinkedIn' },
                ].map(({ ok, label }) => (
                  <span key={label} className={`tag ${ok ? 'tag-success' : 'tag-warning'}`}>
                    {ok ? '✓' : '✗'} {label}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default function CandidateDetailPage() {
  return (
    <Suspense fallback={<div className="loading-container"><div className="loading-spinner" /><p>Loading…</p></div>}>
      <Content />
    </Suspense>
  );
}