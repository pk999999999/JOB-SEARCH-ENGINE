'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { getHealth, setToken, type HealthStatus } from '@/lib/api';
import { LayoutDashboard, Search, Users, FileText, LogOut, Cpu } from 'lucide-react';

const NAV = [
  { href: '/',           label: 'Dashboard',      Icon: LayoutDashboard },
  { href: '/search',     label: 'Search & rank',  Icon: Search },
  { href: '/candidates', label: 'Candidates',     Icon: Users },
  { href: '/jobs',       label: 'Job specs',      Icon: FileText },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => null);
  }, []);

  return (
    <aside className="sidebar" id="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-mark">
          <Cpu size={18} color="var(--cyan)" aria-hidden="true" />
        </div>
        <div>
          <h1>NexusHire</h1>
          <p>Ranker Engine</p>
        </div>
      </div>

      <p className="sidebar-section-label">Menu</p>
      <nav aria-label="Main navigation">
        <ul className="sidebar-nav">
          {NAV.map(({ href, label, Icon }) => (
            <li key={href}>
              <Link
                href={href}
                className={`sidebar-link${pathname === href ? ' active' : ''}`}
              >
                <span className="link-icon" aria-hidden="true"><Icon size={16} /></span>
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <div className="api-status">
          <span className={`status-dot ${health?.status === 'healthy' ? 'healthy' : 'error'}`} />
          <div>
            <div className="status-text-primary">
              {health ? 'API online' : 'Connecting…'}
            </div>
            {health && (
              <div className="status-text-sub">
                {health.candidates_loaded.toLocaleString()} pool · v{health.version}
              </div>
            )}
          </div>
        </div>
        <button
          className="btn btn-secondary btn-sm btn-full"
          onClick={() => { setToken(null); window.location.href = '/login'; }}
        >
          <LogOut size={13} aria-hidden="true" /> Sign out
        </button>
      </div>
    </aside>
  );
}