// Persisted Train-page setup. The user's full run configuration is stashed in
// localStorage under a single versioned key so reopening /train restores the
// last setup. Saved state is MERGED with fresh recommend/defaults on mount, so
// adding new fields here never breaks an old saved blob.

import type { TrainAlgo, TrainHyperparamOverrides, TrainMode } from '../types';

const KEY = 'sumoforge.train.setup.v1';

export type SourceKind = 'current' | 'robot';

/** One opponent's include flag + raw (pre-normalize) weight. */
export interface OppChoice {
  on: boolean;
  weight: number;
}

export interface TrainSetup {
  sourceKind: SourceKind;
  robotId: string;
  algo: TrainAlgo;
  mode: TrainMode;
  baseModelId: string;
  totalSteps: number;
  evalEvery: number;
  startMult: number;
  hyperparams: TrainHyperparamOverrides;
  smoke: boolean;
  // Adaptive opponent weighting: the mix re-weights itself from per-opponent
  // win-rates each eval (guarded by a reserved zoo share + per-opponent cap +
  // EMA), so training auto-focuses on what the model is losing.
  adaptiveOpponents: boolean;
  // Per-opponent include + raw weight, keyed by opponent id. Empty until the
  // opponents list loads and seeds defaults.
  opponents: Record<string, OppChoice>;
}

/** Load the persisted setup, or null if absent/corrupt. Partial is fine — the
 *  caller merges over fresh recommend/defaults. */
export function loadTrainSetup(): Partial<TrainSetup> | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? (parsed as Partial<TrainSetup>) : null;
  } catch {
    return null;
  }
}

export function saveTrainSetup(setup: TrainSetup): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(setup));
  } catch {
    // Storage full / disabled — non-fatal, the page still works in-memory.
  }
}

export function clearTrainSetup(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
