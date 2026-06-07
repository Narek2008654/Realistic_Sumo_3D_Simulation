// <Info topic="track_width" /> — a subtle instrument-style ⓘ affordance that, on
// click, opens an on-brand popover explaining a builder concept in plain words.
//
// Looks like an instrument label, not a generic tooltip: deep panel background,
// hairline border, cyan accent header, Oxanium micro title + mono body. Click-
// outside / Esc / re-click to dismiss; accessible (button + aria, popover role).
// No external deps — the popover is a small absolutely-positioned floating panel.

import { useEffect, useId, useRef, useState } from 'react';
import { HELP } from '../help';

export function Info({
  topic,
  className = '',
  color = 'var(--fg-2)',
  placement = 'bottom',
}: {
  topic: string;
  className?: string;
  // Tint of the ⓘ glyph at rest (e.g. match a legend item colour).
  color?: string;
  // Which side the popover opens. Use 'top' near the bottom of a clipped panel
  // (e.g. the 3D preview legend) so the card isn't cut off.
  placement?: 'top' | 'bottom';
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const panelId = useId();
  const entry = HELP[topic];

  // Dismiss on click-outside or Esc while open.
  useEffect(() => {
    if (!open) return;
    function onPointer(e: PointerEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('pointerdown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // Unknown topic: render nothing rather than a broken control. (Dev guard.)
  if (!entry) {
    if (import.meta.env.DEV) {
      console.warn(`<Info> unknown topic: "${topic}"`);
    }
    return null;
  }

  return (
    <span ref={wrapRef} className={`relative inline-flex ${className}`}>
      <button
        type="button"
        aria-label={`What is ${entry.title}?`}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={(e) => {
          // Don't let the click bubble to a wrapping <label> (which would
          // refocus the input) or to row toggles.
          e.preventDefault();
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className="num inline-flex items-center justify-center"
        style={{
          width: 14,
          height: 14,
          fontSize: 10,
          lineHeight: 1,
          borderRadius: 999,
          border: `1px solid ${open ? 'var(--cyan)' : 'var(--line-2)'}`,
          color: open ? 'var(--cyan)' : color,
          background: open ? 'var(--cyan-glow)' : 'transparent',
          cursor: 'pointer',
          padding: 0,
          transition: 'color .12s ease, border-color .12s ease, background .12s ease',
        }}
      >
        i
      </button>

      {open && (
        <span
          id={panelId}
          role="tooltip"
          className="absolute z-50"
          style={{
            ...(placement === 'top'
              ? { bottom: 'calc(100% + 6px)' }
              : { top: 'calc(100% + 6px)' }),
            left: 0,
            width: 232,
            background: 'var(--bg-1)',
            border: '1px solid var(--line-2)',
            borderRadius: 'var(--radius)',
            boxShadow: 'var(--shadow)',
            padding: '9px 11px',
            cursor: 'default',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* accent hairline along the top — the "live panel" instrument cue */}
          <span
            className="pointer-events-none absolute left-0 right-0 top-0"
            style={{
              height: 1,
              background:
                'linear-gradient(90deg, transparent, var(--cyan), transparent)',
            }}
          />
          <span
            className="micro mb-1 block"
            style={{ color: 'var(--cyan)', fontSize: 10 }}
          >
            {entry.title}
          </span>
          <span
            className="num block text-fg-1"
            style={{ fontSize: 11, lineHeight: 1.5 }}
          >
            {entry.body}
          </span>
        </span>
      )}
    </span>
  );
}
