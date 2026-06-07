// Backend health poller. A single interval drives the engine-status chips in
// the shell. `ok === null` means "not yet probed".

import { useEffect } from 'react';
import { create } from 'zustand';
import { api } from '../api';

interface HealthState {
  ok: boolean | null;
  lastChecked: number | null;
  setOk: (ok: boolean) => void;
}

export const useHealthStore = create<HealthState>((set) => ({
  ok: null,
  lastChecked: null,
  setOk: (ok) => set({ ok, lastChecked: Date.now() }),
}));

const POLL_MS = 5000;

/** Mount once (in the shell) to keep the health store fresh. */
export function useHealthPoll(): void {
  const setOk = useHealthStore((s) => s.setOk);
  useEffect(() => {
    let alive = true;
    const probe = async () => {
      try {
        const r = await api.health();
        if (alive) setOk(r.status === 'ok');
      } catch {
        if (alive) setOk(false);
      }
    };
    probe();
    const t = setInterval(probe, POLL_MS);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [setOk]);
}
