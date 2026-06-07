// Compact labelled form controls with mono numeric readouts.

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

export function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.001,
  unit,
  format = (v) => v.toFixed(3),
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  format?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="micro text-fg-2" style={{ fontSize: 10 }}>
          {label}
        </span>
        <span
          className="num rounded px-1.5 py-0.5"
          style={{
            fontSize: 11,
            color: 'var(--cyan)',
            background: 'var(--bg-2)',
            border: '1px solid var(--line)',
          }}
        >
          {format(value)}
          {unit ? ` ${unit}` : ''}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded"
        style={{ accentColor: 'var(--accent)', background: 'var(--bg-3)' }}
      />
    </label>
  );
}

export function Readout({
  label,
  value,
  tone = 'fg',
}: {
  label: string;
  value: string;
  tone?: 'fg' | 'cyan' | 'accent' | 'win' | 'loss';
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
      <span className="micro text-fg-2" style={{ fontSize: 9 }}>
        {label}
      </span>
      <span className="num" style={{ fontSize: 16, color }}>
        {value}
      </span>
    </div>
  );
}
