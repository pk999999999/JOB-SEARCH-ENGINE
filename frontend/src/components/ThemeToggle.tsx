'use client';
import { useTheme } from '@/components/ThemeProvider';
import { Sun, Moon } from 'lucide-react';

export default function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';

  return (
    <button
      onClick={toggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 14px',
        background: 'var(--s1)',
        border: '1px solid var(--bd2)',
        borderRadius: 99,
        cursor: 'pointer',
        transition: 'all 0.15s',
        userSelect: 'none',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--cyan-bd)';
        e.currentTarget.style.background = 'var(--cyan-bg)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--bd2)';
        e.currentTarget.style.background = 'var(--s1)';
      }}
    >
      {isDark
        ? <Moon size={14} color="var(--cyan)" aria-hidden="true" />
        : <Sun  size={14} color="var(--cyan)" aria-hidden="true" />
      }
      <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--t2)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {isDark ? 'Dark' : 'Light'}
      </span>
      {/* pill track */}
      <div style={{
        width: 34, height: 18,
        background: isDark ? 'var(--cyan)' : 'var(--s3)',
        border: '1px solid var(--bd2)',
        borderRadius: 99,
        position: 'relative',
        transition: 'background 0.2s',
        flexShrink: 0,
      }}>
        <div style={{
          width: 12, height: 12,
          background: '#fff',
          borderRadius: '50%',
          position: 'absolute',
          top: 2,
          left: isDark ? 18 : 2,
          transition: 'left 0.2s',
          boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
        }} />
      </div>
    </button>
  );
}