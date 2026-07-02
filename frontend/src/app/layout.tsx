import type { Metadata } from 'next';
import './globals.css';
import AppContent from '@/components/AppContent';

export const metadata: Metadata = {
  title: 'NexusHire- Candidate Intelligence Engine',
  description: 'AI-powered candidate ranking with semantic matching, fraud detection, and grounded reasoning.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="light">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Syne:wght@500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="orb-1" aria-hidden="true" />
        <div className="orb-2" aria-hidden="true" />
        <AppContent>{children}</AppContent>
      </body>
    </html>
  );
}