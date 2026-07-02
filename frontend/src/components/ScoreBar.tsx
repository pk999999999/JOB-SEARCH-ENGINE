'use client';
import { useEffect, useRef } from 'react';

interface ScoreBarProps {
  label: string;
  value: number;
  color: string;
  animate?: boolean;
}

export default function ScoreBar({ label, value, color, animate = true }: ScoreBarProps) {
  const ref = useRef<HTMLDivElement>(null);
  const pct = Math.min(Math.max(value, 0), 1) * 100;

  useEffect(() => {
    if (!animate || !ref.current) return;
    ref.current.style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (ref.current) ref.current.style.width = `${pct}%`;
    }));
  }, [pct, animate]);

  return (
    <div className="score-bar">
      <div className="score-bar-header">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-value" style={{ color }}>{pct.toFixed(0)}%</span>
      </div>
      <div className="score-bar-track">
        <div
          ref={ref}
          className="score-bar-fill"
          style={{ width: animate ? '0%' : `${pct}%`, background: color, boxShadow: `0 0 6px ${color}60` }}
        />
      </div>
    </div>
  );
}