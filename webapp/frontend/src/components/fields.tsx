// Compact labelled form controls with mono numeric readouts.
import { useEffect, useState } from 'react';
import { Info } from './Info';

export function NumberField({
  label,
  value,
  onChange,
  step = 0.001,
  min,
  max,
  unit,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  unit?: string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="micro text-fg-2" style={{ fontSize: 10 }}>
        {label}
        {unit && <span className="text-fg-2"> · {unit}</span>}
      </span>
      <input
        type="number"
        className="ctl num"
        value={Number.isFinite(value) ? value : ''}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </label>
  );
}

/**
 * Editable numeric box bound to a number value. Keeps a local draft string so
 * the user can type freely (e.g. partial "0.0", "-", or out-of-range values);
 * commits on blur/Enter. Pushes valid intermediate numbers up live so the
 * slider tracks typing. When ``hint`` is set it shows a faint ghost default +
 * a reset chevron to restore it.
 */
function ValueBox({
  value,
  onChange,
  min,
  max,
  unit,
  hint,
  onReset,
  width = 76,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  unit?: string;
  hint?: string;
  onReset?: () => void;
  width?: number;
}) {
  const [draft, setDraft] = useState<string>(String(value));
  const [editing, setEditing] = useState(false);

  // Mirror external changes (slider drag, programmatic load) into the box
  // unless the user is actively typing in it.
  useEffect(() => {
    if (!editing) setDraft(String(value));
  }, [value, editing]);

  function commit() {
    setEditing(false);
    const v = parseFloat(draft);
    if (Number.isFinite(v)) onChange(v);
    else setDraft(String(value));
  }

  return (
    <span className="relative inline-flex items-center gap-1">
      <input
        type="number"
        className="num"
        value={draft}
        step="any"
        min={min}
        max={max}
        onFocus={() => setEditing(true)}
        onChange={(e) => {
          setDraft(e.target.value);
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
        }}
        style={{
          width,
          fontSize: 11,
          color: 'var(--cyan)',
          background: 'var(--bg-2)',
          border: '1px solid var(--line)',
          borderRadius: 'var(--radius)',
          padding: '2px 5px',
          textAlign: 'right',
        }}
      />
      {unit && (
        <span className="micro text-fg-2" style={{ fontSize: 9 }}>
          {unit}
        </span>
      )}
      {hint != null && onReset && (
        <button
          type="button"
          onClick={onReset}
          title={`Reset to recommended ${hint}`}
          className="num"
          style={{
            fontSize: 9,
            lineHeight: 1,
            color: 'var(--fg-2)',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            padding: 0,
          }}
        >
          ⟲
        </button>
      )}
    </span>
  );
}

export function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.001,
  unit,
  format = (v) => v.toFixed(3),
  info,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  format?: (v: number) => string;
  // Optional glossary topic key — adds a subtle ⓘ next to the label that
  // explains this control in plain words (see help.ts / Info.tsx).
  info?: string;
}) {
  // The slider is clamped to [min,max]; the number box accepts any value the
  // user types (clamping only the slider thumb position, not the stored value).
  const clamped = Math.min(max, Math.max(min, value));
  return (
    <label className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
          {label}
          {unit && (
            <span className="text-fg-2" style={{ fontSize: 9 }}>
              · {unit}
            </span>
          )}
          {info && <Info topic={info} />}
        </span>
        <ValueBox value={value} onChange={onChange} min={min} max={max} />
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={clamped}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="h-1.5 w-full cursor-pointer appearance-none rounded"
          style={{ accentColor: 'var(--accent)', background: 'var(--bg-3)' }}
        />
        <span
          className="num text-fg-2"
          style={{ fontSize: 9, minWidth: 56, textAlign: 'right' }}
        >
          {format(value)}
        </span>
      </div>
    </label>
  );
}

export function Readout({
  label,
  value,
  tone = 'fg',
  info,
}: {
  label: string;
  value: string;
  tone?: 'fg' | 'cyan' | 'accent' | 'win' | 'loss';
  info?: string;
}) {
  const color =
    tone === 'cyan'
      ? 'var(--cyan)'
      : tone === 'accent'
        ? 'var(--accent)'
        : tone === 'win'
          ? 'var(--win)'
          : tone === 'loss'
            ? 'var(--loss)'
            : 'var(--fg-0)';
  return (
    <div
      className="flex flex-col gap-1 rounded border px-3 py-2"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 9 }}>
        {label}
        {info && <Info topic={info} />}
      </span>
      <span className="num" style={{ fontSize: 16, color }}>
        {value}
      </span>
    </div>
  );
}
