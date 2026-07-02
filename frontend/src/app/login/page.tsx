'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { login, signup, setToken } from '@/lib/api';
import { Cpu, Eye, EyeOff, AlertCircle, ArrowRight } from 'lucide-react';

export default function LoginPage() {
  const [mode,     setMode]     = useState<'login'|'signup'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPw,   setShowPw]   = useState(false);
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);
  const router = useRouter();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const res = mode === 'login'
        ? await login(username, password)
        : await signup(username, password);
      setToken(res.access_token);
      router.push('/');
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally { setLoading(false); }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px',
      position: 'relative',
      zIndex: 1,
    }}>
      {/* Full-page grid bg already in body::before */}

      <div style={{ width: '100%', maxWidth: 420 }}>

        {/* Logo lockup */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{
            width: 56, height: 56,
            background: 'var(--bg-2)',
            border: '1px solid rgba(0,229,255,0.3)',
            borderRadius: 14,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 16px',
            boxShadow: '0 0 32px rgba(0,229,255,0.12)',
          }}>
            <Cpu size={24} color="var(--cyan)" />
          </div>
          <h1 style={{
            fontFamily: 'var(--font-head)',
            fontSize: '1.75rem',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            background: 'var(--g-main)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: 6,
          }}>
            NexusHire
          </h1>
          <p style={{ fontSize: '0.72rem', color: 'var(--tx-3)', textTransform: 'uppercase', letterSpacing: '0.12em', fontWeight: 700 }}>
            Candidate Ranking Engine
          </p>
        </div>

        {/* Card */}
        <div style={{
          background: 'var(--bg-1)',
          border: '1px solid var(--b1)',
          borderRadius: 20,
          padding: '32px',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 0 0 1px rgba(0,229,255,0.04), 0 24px 64px rgba(0,0,0,0.7)',
        }}>
          {/* top neon line */}
          <div style={{
            position: 'absolute', top: 0, left: '20%', right: '20%',
            height: '1px',
            background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)',
            opacity: 0.5,
          }} />

          {/* Mode tabs */}
          <div style={{
            display: 'flex',
            background: 'var(--bg-2)',
            borderRadius: 10,
            padding: 3,
            marginBottom: 28,
            border: '1px solid var(--b1)',
          }}>
            {(['login', 'signup'] as const).map((m) => (
              <button
                key={m}
                onClick={() => { setMode(m); setError(''); }}
                style={{
                  flex: 1,
                  padding: '7px',
                  border: 'none',
                  borderRadius: 8,
                  fontFamily: 'var(--font-body)',
                  fontSize: '0.83rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  background: mode === m ? 'var(--bg-3)' : 'transparent',
                  color: mode === m ? 'var(--cyan)' : 'var(--tx-3)',
                  boxShadow: mode === m ? '0 0 12px rgba(0,229,255,0.1)' : 'none',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}
              >
                {m === 'login' ? 'Sign in' : 'Sign up'}
              </button>
            ))}
          </div>

          <form onSubmit={submit} noValidate>
            {/* Username */}
            <div className="form-group">
              <label className="form-label" htmlFor="username">Username</label>
              <input
                id="username"
                type="text"
                className="form-input"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="your-username"
                autoComplete="username"
                required
              />
            </div>

            {/* Password */}
            <div className="form-group" style={{ marginBottom: error ? 16 : 24 }}>
              <label className="form-label" htmlFor="password">Password</label>
              <div style={{ position: 'relative' }}>
                <input
                  id="password"
                  type={showPw ? 'text' : 'password'}
                  className="form-input"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  style={{ paddingRight: 44 }}
                  autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  aria-label={showPw ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute', right: 12, top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--tx-3)', display: 'flex', padding: 0,
                    transition: 'color 0.12s',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.color = 'var(--cyan)')}
                  onMouseLeave={e => (e.currentTarget.style.color = 'var(--tx-3)')}
                >
                  {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="error-banner" style={{ marginBottom: 20, fontSize: '0.82rem' }}>
                <AlertCircle size={14} /> {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              className="btn btn-full"
              disabled={loading || !username || !password}
              style={{
                padding: '12px',
                background: 'var(--bg-3)',
                border: '1px solid rgba(0,229,255,0.35)',
                color: loading || !username || !password ? 'var(--tx-3)' : 'var(--cyan)',
                borderColor: loading || !username || !password ? 'var(--b1)' : 'rgba(0,229,255,0.35)',
                fontSize: '0.875rem',
                fontWeight: 700,
                boxShadow: (!loading && username && password) ? '0 0 20px rgba(0,229,255,0.12)' : 'none',
                cursor: loading || !username || !password ? 'not-allowed' : 'pointer',
                opacity: loading || !username || !password ? 0.45 : 1,
                gap: 8,
              }}
            >
              {loading ? (
                <><span className="loading-spinner" style={{ width:14, height:14, borderWidth:2, margin:0 }} /> Authenticating…</>
              ) : (
                <>{mode === 'login' ? 'Sign in' : 'Create account'} <ArrowRight size={15} /></>
              )}
            </button>
          </form>
        </div>

        <p style={{ textAlign: 'center', marginTop: 20, fontSize: '0.78rem', color: 'var(--tx-3)' }}>
          Powered by semantic AI · not keyword search
        </p>
      </div>
    </div>
  );
}