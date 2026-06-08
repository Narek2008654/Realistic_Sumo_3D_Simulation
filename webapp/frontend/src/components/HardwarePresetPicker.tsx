// HardwarePresetPicker — a row of preset chips (novamax + archetypes) that seed
// a HardwareSpec into the builder. Shared by the Opponents authoring page and
// the Hardware builder's CALIBRATE mode so "start from a preset" looks and
// behaves identically in both (DRY).
//
// Controlled + side-effecting: the parent owns the spec; this component fetches
// /api/hardware/presets once and calls onPick(spec) with a deep clone when a
// chip is clicked. The parent decides whether to confirm before replacing an
// edited spec (it knows whether the user has touched the form).

import { useEffect, useState } from 'react';
import { api } from '../api';
import { Info } from './Info';
import type { HardwarePreset, HardwareSpec } from '../types';

export function HardwarePresetPicker({
  onPick,
  label = 'START FROM A PRESET',
  className = '',
}: {
  // Called with a fresh deep clone of the chosen preset's spec.
  onPick: (spec: HardwareSpec, preset: HardwarePreset) => void;
  label?: string;
  className?: string;
}) {
  const [presets, setPresets] = useState<HardwarePreset[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getHardwarePresets()
      .then(setPresets)
      .catch(() => setError('Could not load hardware presets.'));
  }, []);

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
        {label}
        <Info topic="hardware_preset" />
      </span>
      {error ? (
        <span className="num" style={{ fontSize: 10, color: 'var(--warn)' }}>
          {error}
        </span>
      ) : presets.length === 0 ? (
        <span className="num text-fg-2" style={{ fontSize: 10 }}>
          Loading presets…
        </span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {presets.map((p) => (
            <button
              key={p.id}
              type="button"
              title={p.description}
              onClick={() => onPick(structuredClone(p.hardware_spec), p)}
              className="num inline-flex items-center"
              style={{
                fontSize: 10,
                padding: '4px 9px',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--line-2)',
                background: 'var(--bg-2)',
                color: 'var(--fg-1)',
                cursor: 'pointer',
                letterSpacing: '.02em',
              }}
            >
              {p.name}
            </button>
          ))}
        </div>
      )}
      <span className="num text-fg-2" style={{ fontSize: 10, lineHeight: 1.5 }}>
        Click a preset to drop its chassis into the editor — you can still tweak
        every field after.
      </span>
    </div>
  );
}
