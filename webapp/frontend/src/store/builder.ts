// Cross-page handoff store: the Robots page can stash a HardwareSpec here and
// navigate to /hardware, which consumes (and clears) it on mount to seed the
// builder instead of fetching the default spec.

import { create } from 'zustand';
import type { HardwareSpec } from '../types';

interface BuilderState {
  // A spec queued by "Load into Builder", plus the source robot name (for UI).
  pendingSpec: HardwareSpec | null;
  pendingName: string | null;
  loadSpec: (spec: HardwareSpec, name: string) => void;
  // Consume the queued spec exactly once (returns it and clears the slot).
  consume: () => { spec: HardwareSpec; name: string } | null;
}

export const useBuilderStore = create<BuilderState>((set, get) => ({
  pendingSpec: null,
  pendingName: null,
  loadSpec: (spec, name) => set({ pendingSpec: spec, pendingName: name }),
  consume: () => {
    const { pendingSpec, pendingName } = get();
    if (!pendingSpec) return null;
    set({ pendingSpec: null, pendingName: null });
    return { spec: pendingSpec, name: pendingName ?? '' };
  },
}));
