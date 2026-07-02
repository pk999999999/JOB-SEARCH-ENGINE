'use client';
import Link from 'next/link';
import ScoreBar from './ScoreBar';
import type { RankedCandidate } from '@/lib/api';
import { MapPin, Calendar, AlertTriangle, ArrowUpRight, GraduationCap } from 'lucide-react';

const DIM_COLORS: Record<string, string> = {
  career: 'var(--s-career)',
  skills: 'var(--s-skills)',
  experience: 'var(--s-exp)',
  location: 'var(--s-loc)',
  education: 'var(--s-edu)',
  behavioral: 'var(--s-beh)',
};

interface Props {
  candidate: RankedCandidate;
  index: number;
  searchRequiredSkills?: string[];
  searchPreferredSkills?: string[];
}

export default function CandidateCard({ candidate, index, searchRequiredSkills = [], searchPreferredSkills = [] }: Props) {
  const reqSet = new Set(searchRequiredSkills.map(s => s.trim().toLowerCase()));
  const prefSet = new Set(searchPreferredSkills.map(s => s.trim().toLowerCase()));

  const skills = candidate.skills.map(s => {
    const n = s.name.trim().toLowerCase();
    return { ...s, mt: reqSet.has(n) ? 'req' : prefSet.has(n) ? 'pref' : 'none' };
  });

  const candidateNames = new Set(candidate.skills.map(s => s.name.trim().toLowerCase()));
  const missing = searchRequiredSkills.filter(r => !candidateNames.has(r.trim().toLowerCase()));
  const isHoneypot = candidate.score_breakdown?.honeypot_gate === 0;
  const scoreDisplay = isHoneypot ? '0.0' : (candidate.score * 100).toFixed(1);

  return (
    <div
      className="card candidate-card"
      style={{ animationDelay: `${index * 40}ms`, borderColor: isHoneypot ? 'rgba(239,68,68,0.25)' : undefined }}
      id={`candidate-${candidate.candidate_id}`}
    >
      {/* ── Left column ── */}
      <div>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
          <span className="candidate-card-rank">
            Rank &nbsp;
            <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>
              #{candidate.rank}
            </span>
          </span>
          <Link
            href={`/candidates/detail?id=${candidate.candidate_id}`}
            className="btn btn-secondary btn-sm"
            style={{ fontSize: '0.72rem', gap: 4 }}
          >
            View profile <ArrowUpRight size={11} />
          </Link>
        </div>

        <div className="candidate-card-title">
          {candidate.current_title || candidate.headline || candidate.candidate_id}
        </div>

        <div className="candidate-card-meta">
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <MapPin size={12} color="var(--tx-3)" aria-hidden="true" />
            {candidate.location || 'Unknown location'}
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Calendar size={12} color="var(--tx-3)" aria-hidden="true" />
            {candidate.years_of_experience.toFixed(1)} yrs
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <GraduationCap size={12} color="var(--tx-3)" aria-hidden="true" />
            {candidate.education || 'No education text provided'}
          </span>
        </div>

        {/* Honeypot flag */}
        {isHoneypot && (
          <div className="security-alert" role="alert">
            <AlertTriangle size={16} color="#fca5a5" aria-hidden="true" />
            <div>
              <div className="security-alert-title">Profile flagged</div>
              <div className="security-alert-desc">
                Failed honeypot gate — anomalous skill durations or chronological overlaps. Score zeroed.
              </div>
            </div>
          </div>
        )}

        {/* Skills */}
        {(skills.length > 0 || missing.length > 0) && (
          <div className="candidate-card-skills">
            {skills.filter(s => s.mt === 'req').map((s, i) => (
              <span key={`r${i}`} className="tag tag-matched-req" title="Required skill matched">✓ {s.name}</span>
            ))}
            {skills.filter(s => s.mt === 'pref').map((s, i) => (
              <span key={`p${i}`} className="tag tag-matched-pref" title="Preferred skill matched">★ {s.name}</span>
            ))}
            {missing.map((r, i) => (
              <span key={`m${i}`} className="tag tag-missing-req" title="Required skill missing">✗ {r}</span>
            ))}
            {skills.filter(s => s.mt === 'none').slice(0, 5).map((s, i) => (
              <span key={`n${i}`} className="tag">{s.name}</span>
            ))}
            {skills.filter(s => s.mt === 'none').length > 5 && (
              <span className="tag" style={{ opacity: 0.45 }}>+{skills.filter(s => s.mt === 'none').length - 5}</span>
            )}
          </div>
        )}

        {/* AI reasoning */}
        {candidate.reasoning && (
          <div className="candidate-card-reasoning">{candidate.reasoning}</div>
        )}
      </div>

      {/* ── Right column: scores ── */}
      <div className="candidate-card-scores">
        <div className="candidate-card-final-score">
          <div className="score-number" style={isHoneypot ? { color: 'var(--err)', WebkitTextFillColor: 'var(--err)', background: 'none' } : undefined}>
            {scoreDisplay}
          </div>
          <div className="score-label">Match score</div>
        </div>

        {candidate.score_breakdown ? (
          <>
            <ScoreBar label="Career" value={candidate.score_breakdown.career_match} color={DIM_COLORS.career} />
            <ScoreBar label="Skills" value={candidate.score_breakdown.skill_match} color={DIM_COLORS.skills} />
            <ScoreBar label="Experience" value={candidate.score_breakdown.experience_fit} color={DIM_COLORS.experience} />
            <ScoreBar label="Location" value={candidate.score_breakdown.location_fit} color={DIM_COLORS.location} />
            <ScoreBar label="Education" value={candidate.score_breakdown.education_fit} color={DIM_COLORS.education} />

            <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--b1)' }}>
              {[
                { label: 'Coherence', val: candidate.score_breakdown.coherence_factor.toFixed(2) + '×', color: candidate.score_breakdown.coherence_factor < 0.95 ? 'var(--warn)' : 'var(--ok)' },
                { label: 'Behavioral', val: candidate.score_breakdown.behavioral_modifier.toFixed(2) + '×', color: 'var(--cyan)' },
                { label: 'Honeypot', val: isHoneypot ? 'Failed' : 'Passed', color: isHoneypot ? 'var(--err)' : 'var(--ok)' },
              ].map(({ label, val, color }) => (
                <div key={label} className="modifier-row">
                  <span>{label}</span>
                  <span style={{ fontWeight: 700, color, fontFamily: 'var(--font-mono)', fontSize: '0.68rem' }}>{val}</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <ScoreBar label="Overall" value={candidate.score} color={DIM_COLORS.career} />
        )}
      </div>
    </div>
  );
}