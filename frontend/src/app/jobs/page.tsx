'use client';
import { useEffect, useState } from 'react';
import { getJobs, createJob, deleteJob, type JobDescription } from '@/lib/api';
import { Briefcase, Plus, X, AlertCircle, Trash2, Check, FileText } from 'lucide-react';

export default function JobsPage() {
  const [jobs,       setJobs]       = useState<JobDescription[]>([]);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState('');
  const [showForm,   setShowForm]   = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [title,  setTitle]  = useState('');
  const [desc,   setDesc]   = useState('');
  const [req,    setReq]    = useState('');
  const [pref,   setPref]   = useState('');
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setLoading(true);
    try { setJobs((await getJobs()).jobs); }
    catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault(); setSaving(true); setError('');
    try {
      await createJob({
        job_title: title, description: desc,
        required_skills:  req.split(',').map(s => s.trim()).filter(Boolean),
        preferred_skills: pref.split(',').map(s => s.trim()).filter(Boolean),
        experience: { ideal_min_years:5, ideal_max_years:9, absolute_min_years:3, absolute_max_years:15 },
        location: { preferred:['Pune','Noida'], good:['Mumbai','Hyderabad'], acceptable_country:'India' },
        education: { preferred_degrees:[], preferred_fields:[], acceptable_degrees:[] },
      });
      setTitle(''); setDesc(''); setReq(''); setPref('');
      setShowForm(false); load();
    } catch (e: any) { setError(e.message); }
    finally { setSaving(false); }
  };

  const handleDelete = async (id: string) => {
    setDeletingId(id); setError('');
    try { await deleteJob(id); load(); }
    catch (e: any) { setError(e.message); }
    finally { setDeletingId(null); }
  };

  return (
    <>
      <div className="page-header" style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
        <div>
          <h1><Briefcase size={24} color="var(--cyan)" aria-hidden="true" /><span className="title-text">Job specs</span></h1>
          <p>Manage job descriptions used for AI candidate ranking</p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => { setShowForm(!showForm); setError(''); }}
        >
          {showForm ? <><X size={15} /> Cancel</> : <><Plus size={15} /> New job</>}
        </button>
      </div>

      {/* ── Create form ── */}
      {showForm && (
        <div className="card" style={{ marginBottom: 24 }}>
          <h3 className="section-title" style={{ marginBottom: 20 }}>
            <Plus size={13} color="var(--cyan)" /> New job description
          </h3>
          <form onSubmit={handleCreate}>
            <div className="form-group">
              <label className="form-label" htmlFor="jt">Job title</label>
              <input id="jt" className="form-input" value={title}
                onChange={e => setTitle(e.target.value)} placeholder="ML Platform Engineer" required />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="jd">Description</label>
              <textarea id="jd" className="form-textarea" rows={3} value={desc}
                onChange={e => setDesc(e.target.value)} placeholder="Describe the role…" />
            </div>
            <div className="search-form-row">
              <div className="form-group">
                <label className="form-label" htmlFor="jr">Required skills</label>
                <textarea id="jr" className="form-textarea" rows={2} value={req}
                  onChange={e => setReq(e.target.value)} placeholder="python, pytorch, kafka…" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="jp">Preferred skills</label>
                <textarea id="jp" className="form-textarea" rows={2} value={pref}
                  onChange={e => setPref(e.target.value)} placeholder="docker, k8s, aws…" />
              </div>
            </div>
            <button type="submit" className="btn btn-primary" disabled={saving || !title.trim()}>
              {saving
                ? <><span className="loading-spinner" style={{ width:13,height:13,borderWidth:2,margin:0 }} /> Saving…</>
                : <><Check size={14} /> Create job</>
              }
            </button>
          </form>
        </div>
      )}

      {/* ── Error ── */}
      {error && <div className="error-banner" role="alert"><AlertCircle size={14} /> {error}</div>}

      {/* ── List ── */}
      {loading ? (
        <div className="loading-container"><div className="loading-spinner" /><p>Loading…</p></div>
      ) : (
        <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
          {jobs.map(job => (
            <div key={job.id} className="card" id={`job-${job.id}`}
              style={{ borderLeft: job.id === 'default' ? '2px solid var(--cyan)' : '2px solid transparent' }}
            >
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', gap:16 }}>
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:6 }}>
                    <h3 style={{ fontSize:'0.95rem', fontWeight:700, fontFamily:'var(--font-head)' }}>
                      {job.job_title}
                    </h3>
                    {job.id === 'default' && (
                      <span style={{
                        fontSize:'0.6rem', fontWeight:800, padding:'2px 8px',
                        background:'rgba(0,229,255,0.08)', border:'1px solid rgba(0,229,255,0.25)',
                        borderRadius:99, color:'var(--cyan)',
                        textTransform:'uppercase', letterSpacing:'0.06em',
                      }}>Default</span>
                    )}
                  </div>
                  {job.description && (
                    <p style={{ color:'var(--tx-2)', fontSize:'0.83rem', lineHeight:1.6, marginBottom:12 }}>
                      {job.description.length > 200 ? job.description.substring(0,200)+'…' : job.description}
                    </p>
                  )}
                  <div style={{ display:'flex', flexWrap:'wrap', gap:5 }}>
                    {job.required_skills.slice(0,8).map((s,i) => (
                      <span key={i} className="tag">{s}</span>
                    ))}
                    {job.required_skills.length > 8 && (
                      <span style={{ fontSize:'0.7rem', color:'var(--tx-3)', alignSelf:'center' }}>
                        +{job.required_skills.length - 8} more
                      </span>
                    )}
                  </div>
                </div>
                {job.id !== 'default' && (
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDelete(job.id)}
                    disabled={deletingId === job.id}
                    style={{ flexShrink:0 }}
                  >
                    {deletingId === job.id
                      ? <span className="loading-spinner" style={{ width:12,height:12,borderWidth:2,margin:0 }} />
                      : <><Trash2 size={12} /> Delete</>
                    }
                  </button>
                )}
              </div>
            </div>
          ))}

          {jobs.length === 0 && (
            <div className="empty-state">
              <div className="icon" style={{ color:'var(--tx-3)' }}><FileText size={44} /></div>
              <h3>No job specs yet</h3>
              <p>Create one to start ranking candidates.</p>
            </div>
          )}
        </div>
      )}
    </>
  );
}