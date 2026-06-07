// Left-nav app shell: fixed rail (wordmark + nav + engine-status footer),
// top bar (section title + backend chip), engineering-grid backdrop.
import { motion } from 'framer-motion';
import type { ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useHealthPoll, useHealthStore } from '../store/health';
import { StatusDot } from './ui';

interface NavItem {
  to: string;
  label: string;
  icon: ReactNode;
}

// Minimal stroked glyphs (no icon dep) — technical, monoline.
const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

const NAV: NavItem[] = [
  {
    to: '/hardware',
    label: 'Hardware',
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" {...stroke}>
        <rect x="6" y="6" width="12" height="12" rx="1" />
        <path d="M3 9h3M3 15h3M18 9h3M18 15h3M9 3v3M15 3v3M9 18v3M15 18v3" />
      </svg>
    ),
  },
  {
    to: '/train',
    label: 'Train',
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" {...stroke}>
        <path d="M4 19V5M4 19h16M8 16l3-5 3 3 5-7" />
      </svg>
    ),
  },
  {
    to: '/arena',
    label: 'Arena',
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" {...stroke}>
        <circle cx="12" cy="12" r="8" />
        <circle cx="12" cy="12" r="2.5" />
      </svg>
    ),
  },
  {
    to: '/opponents',
    label: 'Opponents',
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" {...stroke}>
        <circle cx="8" cy="9" r="3" />
        <circle cx="16" cy="9" r="3" />
        <path d="M3 20c0-3 2.5-5 5-5s5 2 5 5M13 20c0-3 2.5-5 5-5" />
      </svg>
    ),
  },
  {
    to: '/models',
    label: 'Models',
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" {...stroke}>
        <path d="M4 7l8-4 8 4-8 4-8-4z" />
        <path d="M4 7v10l8 4 8-4V7M12 11v10" />
      </svg>
    ),
  },
];

function EngineStatus() {
  const ok = useHealthStore((s) => s.ok);
  const status = ok === null ? 'idle' : ok ? 'ok' : 'down';
  const label = ok === null ? 'Probing' : ok ? 'Connected' : 'Offline';
  return (
    <div
      className="mx-3 mb-3 flex items-center gap-2 rounded border px-3 py-2"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <StatusDot status={status} pulse={ok === true} />
      <div className="flex flex-col leading-tight">
        <span className="micro text-fg-2" style={{ fontSize: 9 }}>
          SUMO ENGINE
        </span>
        <span
          className="micro"
          style={{
            color:
              status === 'ok'
                ? 'var(--cyan)'
                : status === 'down'
                  ? 'var(--loss)'
                  : 'var(--fg-1)',
            fontSize: 11,
          }}
        >
          {label}
        </span>
      </div>
    </div>
  );
}

function BackendChip() {
  const ok = useHealthStore((s) => s.ok);
  const status = ok === null ? 'idle' : ok ? 'ok' : 'down';
  return (
    <span
      className="inline-flex items-center gap-2 rounded border px-2.5 py-1"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-1)' }}
    >
      <StatusDot status={status} pulse={ok === true} />
      <span className="micro" style={{ fontSize: 10 }}>
        BACKEND
      </span>
      <span
        className="num text-fg-2"
        style={{ fontSize: 10 }}
      >
        127.0.0.1:8000
      </span>
    </span>
  );
}

const TITLES: Record<string, string> = {
  '/hardware': 'Hardware Builder',
  '/train': 'Training',
  '/arena': 'Arena',
  '/opponents': 'Opponents',
  '/models': 'Model Registry',
};

export function AppShell({ children }: { children: ReactNode }) {
  useHealthPoll();
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? 'SumoForge';

  return (
    <div className="relative min-h-screen">
      <div className="app-backdrop" />
      <div className="app-grain" />
      <div className="app-vignette" />

      <div className="relative z-10 flex min-h-screen">
        {/* Left rail */}
        <aside
          className="fixed inset-y-0 left-0 flex w-[228px] flex-col border-r"
          style={{ background: 'var(--bg-1)', borderColor: 'var(--line)' }}
        >
          <div className="px-5 pb-5 pt-6">
            <h1
              className="font-display text-[22px] font-bold tracking-wide"
              style={{ color: 'var(--fg-0)' }}
            >
              SUMO<span style={{ color: 'var(--accent)' }}>FORGE</span>
            </h1>
            <div className="micro mt-1 text-fg-2" style={{ fontSize: 9 }}>
              DOHYO MISSION CONSOLE
            </div>
          </div>

          <nav className="flex-1 px-2">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} className="block">
                {({ isActive }) => (
                  <div
                    className="relative my-0.5 flex items-center gap-3 rounded px-3 py-2.5 transition-colors"
                    style={{
                      background: isActive
                        ? 'var(--accent-glow)'
                        : 'transparent',
                      color: isActive ? 'var(--fg-0)' : 'var(--fg-1)',
                    }}
                  >
                    {isActive && (
                      <motion.span
                        layoutId="nav-active-bar"
                        className="absolute left-0 top-1/2 h-6 w-[2px] -translate-y-1/2"
                        style={{
                          background: 'var(--accent)',
                          boxShadow: '0 0 10px var(--accent-glow)',
                        }}
                      />
                    )}
                    <span
                      style={{
                        color: isActive ? 'var(--accent)' : 'var(--fg-2)',
                      }}
                    >
                      {item.icon}
                    </span>
                    <span
                      className="font-display text-[13px] uppercase tracking-[.08em]"
                      style={{ fontWeight: isActive ? 600 : 400 }}
                    >
                      {item.label}
                    </span>
                  </div>
                )}
              </NavLink>
            ))}
          </nav>

          <EngineStatus />
        </aside>

        {/* Content area */}
        <div className="ml-[228px] flex min-h-screen flex-1 flex-col">
          <header
            className="sticky top-0 z-20 flex h-14 items-center justify-between border-b px-6 backdrop-blur"
            style={{
              background: 'rgba(8,11,15,.7)',
              borderColor: 'var(--line)',
            }}
          >
            <div className="flex items-baseline gap-3">
              <h2 className="font-display text-[18px] font-semibold uppercase tracking-[.06em]">
                {title}
              </h2>
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                {pathname}
              </span>
            </div>
            <BackendChip />
          </header>

          <main className="flex-1 p-6">{children}</main>
        </div>
      </div>
    </div>
  );
}
