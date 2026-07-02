'use client';
import { usePathname } from 'next/navigation';
import { ThemeProvider } from '@/components/ThemeProvider';
import Sidebar from '@/components/Sidebar';
import ThemeToggle from '@/components/ThemeToggle';

export default function AppContent({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLogin = pathname === '/login';

  if (isLogin) {
    return (
      <ThemeProvider>
        <main>{children}</main>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          {/* Theme toggle pinned to top-right of every page */}
          <div style={{
            display: 'flex',
            justifyContent: 'flex-end',
            marginBottom: 'var(--sp4)',
          }}>
            <ThemeToggle />
          </div>
          {children}
        </main>
      </div>
    </ThemeProvider>
  );
}