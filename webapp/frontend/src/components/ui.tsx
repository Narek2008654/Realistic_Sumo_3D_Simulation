// Shared instrument-panel UI primitives: corner brackets, panels, status dots.
import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

/** L-shaped corner ticks for the instrument look (on hero panels/viewport). */
export function CornerTicks({ color = 'var(--line-2)' }: { color?: string }) {
  const base =
    'pointer-events-none absolute h-3 w-3 border-[var(--line-2)]';
  return (
    <>
      <span
        className={`${base} left-0 top-0 border-l border-t`}
        style={{ borderColor: color }}
      />
      <span
        className={`${base} right-0 top-0 border-r border-t`}
        style={{ borderColor: color }}
      />
      <span
        className={`${base} bottom-0 left-0 border-b border-l`}
        style={{ borderColor: color }}
      />
      <span
        className={`${base} bottom-0 right-0 border-b border-r`}
        style={{ borderColor: color }}
      />
    </>
  );
}

export function Panel({
  title,
  right,
  live = false,
  ticks = false,
  className = '',
  bodyClassName = '',
  children,
}: {
  title?: string;
  right?: ReactNode;
  live?: boolean;
  ticks?: boolean;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  return (
    <div className={`panel ${live ? 'panel-live' : ''} ${className}`}>
      {ticks && <CornerTicks />}
      {title && (
        <div className="panel-head">
          <span className="micro text-fg-1">{title}</span>
          {right}
        </div>
      )}
      <div className={bodyClassName || 'p-4'}>{children}</div>
    </div>
  );
}

const STATUS_COLOR: Record<string, string> = {
  ok: 'var(--cyan)',
  win: 'var(--win)',
  down: 'var(--loss)',
  loss: 'var(--loss)',
  warn: 'var(--warn)',
  idle: 'var(--idle)',
  accent: 'var(--accent)',
};

export function StatusDot({
  status,
  pulse = false,
}: {
  status: keyof typeof STATUS_COLOR | string;
  pulse?: boolean;
}) {
  const color = STATUS_COLOR[status] ?? 'var(--idle)';
  return (
    <span className="relative inline-flex h-2 w-2">
      {pulse && (
        <span
          className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
          style={{ background: color }}
        />
      )}
      <span
        className="relative inline-flex h-2 w-2 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
    </span>
  );
}

export function StatusPill({
  status,
  label,
  pulse = false,
}: {
  status: string;
  label: string;
  pulse?: boolean;
}) {
  const color = STATUS_COLOR[status] ?? 'var(--idle)';
  return (
    <span
      className="inline-flex items-center gap-2 rounded border px-2 py-1"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <StatusDot status={status} pulse={pulse} />
      <span
        className="micro"
        style={{ color, letterSpacing: '.1em' }}
      >
        {label}
      </span>
    </span>
  );
}

/** Staggered panel reveal wrapper (translateY + fade). */
export function Reveal({
  index = 0,
  className = '',
  children,
}: {
  index?: number;
  className?: string;
  children: ReactNode;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.35,
        delay: index * 0.05,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      {children}
    </motion.div>
  );
}
