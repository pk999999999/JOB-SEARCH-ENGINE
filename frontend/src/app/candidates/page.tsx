'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Users, Upload, Search, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';
import { getCandidates, uploadCandidate, type CandidateListResponse } from '@/lib/api';

const buildPages = (cur: number, total: number): (number | '…')[] => {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  if (cur <= 4) return [1, 2, 3, 4, 5, '…', total];
  if (cur >= total - 3) return [1, '…', total - 4, total - 3, total - 2, total - 1, total];
  return [1, '…', cur - 1, cur, cur + 1, '…', total];
};

export default function CandidatesPage() {
  const [data, setData] = useState<CandidateListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [page, setPage] = useState(1);
  const [location, setLocation] = useState('');
  const [skill, setSkill] = useState('');
  const [minExp, setMinExp] = useState('');
  const [maxExp, setMaxExp] = useState('');

  const fetchData = async (p = page) => {
    setLoading(true); setError('');
    try {
      const r = await getCandidates({
        page: p, page_size: 20,
        location: location || undefined, skill: skill || undefined,
        min_experience: minExp ? Number(minExp) : undefined,
        max_experience: maxExp ? Number(maxExp) : undefined,
      });
      setData(r); setPage(p);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchData(1); }, []); // eslint-disable-line

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) { setError('PDF only'); return; }
    setUploading(true); setError('');
    try { await uploadCandidate(file); await fetchData(1); }
    catch (err: any) { setError(err.message || 'Upload failed'); }
    finally { setUploading(false); e.target.value = ''; }
  };

  return (
    <>
      {/* ── Header ── */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1><Users size={24} color="var(--cyan)" aria-hidden="true" /><span className="title-text">Candidates</span></h1>
          <p>Browse and filter the candidate database</p>
        </div>
        <div>
          <label
            htmlFor="resume-upload"
            className="btn btn-primary"
            style={{ cursor: uploading ? 'not-allowed' : 'pointer', opacity: uploading ? 0.5 : 1 }}
          >
            {uploading
              ? <><span className="loading-spinner" style={{ width: 14, height: 14, borderWidth: 2, margin: 0 }} /> Uploading…</>
              : <><Upload size={15} aria-hidden="true" /> Upload resume</>
            }
          </label>
          <input id="resume-upload" type="file" accept=".pdf" style={{ display: 'none' }}
            onChange={handleUpload} disabled={uploading} />
        </div>
      </div>

      {/* ── Filters ── */}
      <div className="card" style={{ marginBottom: 24 }}>
        <form
          onSubmit={e => { e.preventDefault(); fetchData(1); }}
          style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}
        >
          {[
            { label: 'Location', id: 'f-loc', value: location, set: setLocation, placeholder: 'Pune', flex: '1 1 150px' },
            { label: 'Skill', id: 'f-sk', value: skill, set: setSkill, placeholder: 'python', flex: '1 1 150px' },
            { label: 'Min exp', id: 'f-mn', value: minExp, set: setMinExp, placeholder: '0', flex: '0 1 90px', type: 'number' },
            { label: 'Max exp', id: 'f-mx', value: maxExp, set: setMaxExp, placeholder: '20', flex: '0 1 90px', type: 'number' },
          ].map(({ label, id, value, set, placeholder, flex, type }) => (
            <div key={id} className="form-group" style={{ flex, marginBottom: 0 }}>
              <label className="form-label" htmlFor={id}>{label}</label>
              <input id={id} className="form-input" type={type ?? 'text'}
                value={value} onChange={e => set(e.target.value)} placeholder={placeholder} />
            </div>
          ))}
          <button type="submit" className="btn btn-primary" style={{ height: 40, alignSelf: 'flex-end' }}>
            <Search size={14} aria-hidden="true" /> Filter
          </button>
        </form>
      </div>

      {/* ── Error ── */}
      {error && <div className="error-banner" role="alert"><AlertCircle size={14} /> {error}</div>}

      {/* ── Loading ── */}
      {loading && <div className="loading-container"><div className="loading-spinner" /><p>Loading candidates…</p></div>}

      {/* ── Table ── */}
      {data && !loading && (
        <>
          <div style={{ marginBottom: 12, color: 'var(--tx-3)', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
            {data.candidates.length.toLocaleString()} / {data.total.toLocaleString()} results
            <span style={{ margin: '0 8px', opacity: 0.3 }}>·</span>
            page {data.page} of {data.total_pages}
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Location</th>
                  <th>Exp</th>
                  <th>Skills</th>
                  <th style={{ width: 80 }}></th>
                </tr>
              </thead>
              {/* ✅ NEW SAFE/CRASH-PROOF CODE */}
              <tbody>
                {(data?.candidates || []).map(c => (
                  <tr key={c.candidate_id}>
                    <td>
                      <div style={{ fontWeight: 700, color: 'var(--tx-1)' }}>{c.current_title || '—'}</div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--tx-3)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                        {c.headline?.substring(0, 52) || c.candidate_id}
                      </div>
                    </td>
                    <td style={{ color: 'var(--tx-2)' }}>{c.location || '—'}</td>
                    <td style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontWeight: 700, whiteSpace: 'nowrap' }}>
                      {typeof c.years_of_experience === 'number' ? c.years_of_experience.toFixed(1) : '0.0'}y
                    </td>
                    <td>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {c.skills?.slice(0, 3).map((s: any, i: number) => (
                          <span key={i} className="tag" style={{ fontSize: '0.65rem' }}>
                            {typeof s === 'object' ? s?.name : s}
                          </span>
                        ))}
                        {(c.skills?.length || 0) > 3 && (
                          <span style={{ fontSize: '0.65rem', color: 'var(--tx-3)', alignSelf: 'center' }}>
                            +{(c.skills?.length || 0) - 3}
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <Link href={`/candidates/detail?id=${c.candidate_id}`} className="btn btn-secondary btn-sm">
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {data.total_pages > 1 && (
            <nav className="pagination" aria-label="Pagination">
              <button className="pagination-btn" onClick={() => fetchData(page - 1)} disabled={page <= 1} aria-label="Previous">
                <ChevronLeft size={14} />
              </button>
              {buildPages(page, data.total_pages).map((p, i) =>
                p === '…'
                  ? <span key={`e${i}`} className="pagination-ellipsis">…</span>
                  : <button key={p} className={`pagination-btn${page === p ? ' active' : ''}`}
                    onClick={() => fetchData(p as number)} aria-current={page === p ? 'page' : undefined}>
                    {p}
                  </button>
              )}
              <button className="pagination-btn" onClick={() => fetchData(page + 1)} disabled={page >= data.total_pages} aria-label="Next">
                <ChevronRight size={14} />
              </button>
            </nav>
          )}
        </>
      )}

      {/* ── Empty ── */}
      {data && data.candidates.length === 0 && !loading && (
        <div className="empty-state">
          <div className="icon" style={{ color: 'var(--tx-3)' }}><Users size={48} /></div>
          <h3>No candidates found</h3>
          <p>Try different filters or upload resume data.</p>
        </div>
      )}
    </>
  );
}