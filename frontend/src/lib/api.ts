const API_BASE = process.env.NEXT_PUBLIC_API_URL !== undefined && process.env.NEXT_PUBLIC_API_URL !== ''
  ? process.env.NEXT_PUBLIC_API_URL
  : (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000');

export function getToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('access_token');
  }
  return null;
}

export function setToken(token: string | null) {
  if (typeof window !== 'undefined') {
    if (token) {
      localStorage.setItem('access_token', token);
    } else {
      localStorage.removeItem('access_token');
    }
  }
}

/**
 * Generic fetch wrapper with error handling and JWT injection.
 */
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  
  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(url, {
    ...options,
    headers: {
      ...headers,
      ...options.headers,
    },
  });

  if (res.status === 401) {
    setToken(null);
    if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
      window.location.href = '/login';
    }
  }

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const url = `${API_BASE}/api/auth/token`;
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData.toString(),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Login failed: ${res.status}`);
  }

  return res.json();
}

export async function signup(username: string, password: string): Promise<{ access_token: string }> {
  const url = `${API_BASE}/api/auth/signup`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Sign up failed: ${res.status}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Skill {
  name: string;
  endorsements: number;
  duration_months: number;
  proficiency: string;
}

export interface CareerEntry {
  title: string;
  company: string;
  start_date: string;
  end_date: string;
  duration_months: number;
  is_current: boolean;
  description: string;
}

export interface EducationEntry {
  degree: string;
  field_of_study: string;
  institution: string;
  year: number;
}

export interface BehavioralSignals {
  last_active_date: string;
  recruiter_response_rate: number;
  avg_response_time_hours: number;
  interview_completion_rate: number;
  offer_acceptance_rate: number;
  verified_email: boolean;
  verified_phone: boolean;
  linkedin_connected: boolean;
}

export interface ScoreBreakdown {
  career_match: number;
  skill_match: number;
  experience_fit: number;
  location_fit: number;
  education_fit: number;
  rule_adjustments: number;
  base_score: number;
  coherence_factor: number;
  behavioral_modifier: number;
  honeypot_gate: number;
  final_score: number;
}

export interface RankedCandidate {
  candidate_id: string;
  rank: number;
  score: number;
  reasoning: string;
  headline: string;
  current_title: string;
  location: string;
  years_of_experience: number;
  skills: Skill[];
  education: string;
  score_breakdown?: ScoreBreakdown;
}

export interface RankResponse {
  job_title: string;
  total_candidates: number;
  ranked_count: number;
  elapsed_seconds: number;
  candidates: RankedCandidate[];
}

export interface CandidateDetail {
  candidate_id: string;
  headline: string;
  summary: string;
  current_title: string;
  location: string;
  years_of_experience: number;
  skills: Skill[];
  career_history: CareerEntry[];
  education: EducationEntry[];
  behavioral_signals: BehavioralSignals;
  salary_min: number | null;
  salary_max: number | null;
}

export interface CandidateListResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  candidates: CandidateDetail[];
}

export interface JobDescription {
  id: string;
  job_title: string;
  description: string;
  required_skills: string[];
  preferred_skills: string[];
  experience: {
    ideal_min_years: number;
    ideal_max_years: number;
    absolute_min_years: number;
    absolute_max_years: number;
  };
  location: {
    preferred: string[];
    good: string[];
    acceptable_country: string;
  };
  education: {
    preferred_degrees: string[];
    preferred_fields: string[];
    acceptable_degrees: string[];
  };
}

export interface Stats {
  total_candidates: number;
  avg_experience_years: number;
  top_locations: { location: string; count: number }[];
  total_jobs: number;
  skill_distribution: { skill: string; count: number }[];
}

export interface HealthStatus {
  status: string;
  version: string;
  candidates_loaded: number;
  artifacts_available: boolean;
}

// ---------------------------------------------------------------------------
// API Functions
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>('/api/health');
}

export async function getStats(): Promise<Stats> {
  return apiFetch<Stats>('/api/stats');
}

export async function rankCandidates(request: {
  job_title: string;
  description?: string;
  required_skills: string[];
  preferred_skills: string[];
  top_k?: number;
  experience?: {
    ideal_min_years?: number;
    ideal_max_years?: number;
  };
  location?: {
    preferred?: string[];
    good?: string[];
    acceptable_country?: string;
  };
}): Promise<RankResponse> {
  return apiFetch<RankResponse>('/api/rank', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getCandidates(params: {
  page?: number;
  page_size?: number;
  location?: string;
  min_experience?: number;
  max_experience?: number;
  skill?: string;
}): Promise<CandidateListResponse> {
  const searchParams = new URLSearchParams();
  if (params.page !== undefined) searchParams.set('page', String(params.page));
  if (params.page_size !== undefined) searchParams.set('page_size', String(params.page_size));
  if (params.location) searchParams.set('location', params.location);
  if (params.min_experience !== undefined)
    searchParams.set('min_experience', String(params.min_experience));
  if (params.max_experience !== undefined)
    searchParams.set('max_experience', String(params.max_experience));
  if (params.skill) searchParams.set('skill', params.skill);

  return apiFetch<CandidateListResponse>(`/api/candidates?${searchParams}`);
}

export async function getCandidate(id: string): Promise<CandidateDetail> {
  return apiFetch<CandidateDetail>(`/api/candidates/${id}`);
}

export async function getJobs(): Promise<{ total: number; jobs: JobDescription[] }> {
  return apiFetch<{ total: number; jobs: JobDescription[] }>('/api/jobs');
}

export async function createJob(
  job: Omit<JobDescription, 'id'>
): Promise<JobDescription> {
  return apiFetch<JobDescription>('/api/jobs', {
    method: 'POST',
    body: JSON.stringify(job),
  });
}

export async function deleteJob(id: string): Promise<void> {
  await apiFetch(`/api/jobs/${id}`, { method: 'DELETE' });
}

export async function uploadCandidate(file: File): Promise<CandidateDetail> {
  const url = `${API_BASE}/api/candidates/upload`;
  const token = getToken();
  
  const formData = new FormData();
  formData.append('file', file);
  
  const headers: Record<string, string> = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Upload failed: ${res.status}`);
  }
  
  return res.json();
}
