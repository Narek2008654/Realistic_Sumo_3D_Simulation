// OPPONENTS — author custom opponents in two stages (build the FULL HARDWARE,
// then the behaviour LOGIC), list/delete them, test one in a tiny battle, and
// make them selectable in the Arena (via the side-B picker).
//
// Two columns:
//  • LEFT  — CREATE panel, a two-stage flow:
//      Stage 1 · HARDWARE — the full hardware editor (the same chassis / CoM /
//        wedge / drivetrain / dohyo / distance + line sensor controls the robot
//        Hardware builder uses, shared via <HardwareForm>), seeded from
//        /api/hardware/default (or loaded from a saved robot), with a live 3D
//        RobotPreview of the opponent being built. "NEXT: BEHAVIOR →".
//      Stage 2 · LOGIC — the rule builder (ordered WHEN→DO rules with a
//        pragmatic condition editor: predicate + ALL/ANY combine + NOT + timer),
//        a DEFAULT action, a debounced live /validate readout, then NAME + SAVE.
//        "← BACK" returns to Stage 1.
//  • RIGHT — the saved-opponent ROSTER (instrument cards: name, #rules,
//    created_at, DELETE + TEST), plus the TEST battle result (1 round, real
//    physics) replayed in the reused DOHYO-CAM TrajectoryPlayer.
//
// The custom opponent now fights on the hardware built here (the backend spawns
// it on its own chassis + motors), so the full spec the user authors is used in
// sim — and its behaviour rules drive what it does.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusPill } from '../components/ui';
import { Info } from '../components/Info';
import { HardwareForm } from '../components/HardwareForm';
import { HardwarePresetPicker } from '../components/HardwarePresetPicker';
import { RobotPreview } from '../components/RobotPreview';
import { TrajectoryPlayer } from '../components/TrajectoryPlayer';
import { HELP } from '../help';
import type {
  BattleResult,
  CustomOpponentSummary,
  Geometry,
  HardwareSpec,
  ModelCard,
  OpponentAction,
  OpponentBehavior,
  OpponentCond,
  OpponentDsl,
  OpponentPredicate,
  OpponentRule,
  RobotSummary,
  TrainOpponent,
} from '../types';

// Which behavior source Stage 2 is authoring: a built-in zoo controller or a
// user-authored rule DSL.
type BehaviorSource = 'zoo' | 'dsl';

// Friendly display name + help topic for a zoo controller id. Falls back to the
// raw id (and no ⓘ) for any future id we don't have copy for yet.
function zooInfoTopic(id: string): string | null {
  const topic = `zoo_${id}`;
  return HELP[topic] ? topic : null;
}

const VALIDATE_DEBOUNCE_MS = 350;

const ACTIONS: OpponentAction[] = [
  'forward',
  'reverse',
  'spin_left',
  'spin_right',
  'arc_left',
  'arc_right',
  'stop',
];

const PREDICATES: OpponentPredicate[] = [
  'front_hit',
  'left_hit',
  'right_hit',
  'side_left_hit',
  'side_right_hit',
  'edge_left',
  'edge_right',
  'no_target',
];

// Plain-language labels for the dropdowns (the ⓘ carries the full explanation).
const ACTION_LABEL: Record<OpponentAction, string> = {
  forward: 'Drive forward',
  reverse: 'Reverse',
  spin_left: 'Spin left',
  spin_right: 'Spin right',
  arc_left: 'Arc left',
  arc_right: 'Arc right',
  stop: 'Stop',
};

const PREDICATE_LABEL: Record<OpponentPredicate, string> = {
  front_hit: 'Enemy ahead (front)',
  left_hit: 'Enemy front-left',
  right_hit: 'Enemy front-right',
  side_left_hit: 'Enemy on left side',
  side_right_hit: 'Enemy on right side',
  edge_left: 'Left wheel near edge',
  edge_right: 'Right wheel near edge',
  no_target: 'Lost the enemy',
};

const PREDICATE_INFO: Record<OpponentPredicate, string> = {
  front_hit: 'pred_front_hit',
  left_hit: 'pred_left_hit',
  right_hit: 'pred_right_hit',
  side_left_hit: 'pred_side_left_hit',
  side_right_hit: 'pred_side_right_hit',
  edge_left: 'pred_edge_left',
  edge_right: 'pred_edge_right',
  no_target: 'pred_no_target',
};

type Combine = 'all' | 'any';

// ---------------------------------------------------------------------------
// Builder model: a rule is edited as a flat, pragmatic struct, then compiled to
// the backend condition tree on validate/save. `preds` is the chosen predicate
// set; `combine` only matters with >1; `not` inverts the whole condition;
// `timer` (when set) replaces the predicate condition with {timer:{every:N}}.
// ---------------------------------------------------------------------------
interface RuleDraft {
  key: string;
  preds: OpponentPredicate[];
  combine: Combine;
  not: boolean;
  useTimer: boolean;
  timerEvery: number;
  action: OpponentAction;
}

let _seq = 0;
function newRule(partial?: Partial<RuleDraft>): RuleDraft {
  return {
    key: `r${_seq++}`,
    preds: ['front_hit'],
    combine: 'all',
    not: false,
    useTimer: false,
    timerEvery: 50,
    action: 'forward',
    ...partial,
  };
}

/** Compile one draft to a backend condition node. */
function compileCond(r: RuleDraft): OpponentCond {
  if (r.useTimer) {
    return { timer: { every: Math.max(1, Math.round(r.timerEvery)) } };
  }
  let cond: OpponentCond;
  if (r.preds.length <= 1) {
    cond = r.preds[0] ?? 'no_target';
  } else {
    cond =
      r.combine === 'all'
        ? { all: [...r.preds] }
        : { any: [...r.preds] };
  }
  return r.not ? { not: cond } : cond;
}

function compileDsl(rules: RuleDraft[], def: OpponentAction): OpponentDsl {
  return {
    rules: rules.map(
      (r): OpponentRule => ({ when: compileCond(r), do: r.action }),
    ),
    default: def,
  };
}

// Count rules without instantiating the full record (list rows omit the DSL).
function ruleCountLabel(n: number | null): string {
  if (n == null) return '— rules';
  return `${n} rule${n === 1 ? '' : 's'}`;
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ===========================================================================
// Page
// ===========================================================================
export default function Opponents() {
  // Roster
  const [roster, setRoster] = useState<CustomOpponentSummary[]>([]);
  const [ruleCounts, setRuleCounts] = useState<Record<string, number>>({});
  const [rosterError, setRosterError] = useState<string | null>(null);

  // Builder — two-stage: HARDWARE first, then LOGIC.
  const [stage, setStage] = useState<'hardware' | 'logic'>('hardware');
  const [name, setName] = useState('');

  // Stage 2 · BEHAVIOR — either a built-in zoo controller or a custom rule DSL.
  const [behaviorSource, setBehaviorSource] = useState<BehaviorSource>('zoo');
  const [zooId, setZooId] = useState('');
  const [zooList, setZooList] = useState<TrainOpponent[]>([]);

  // Custom-rules (DSL) builder state.
  const [rules, setRules] = useState<RuleDraft[]>([
    newRule({ preds: ['front_hit'], action: 'forward' }),
    newRule({ preds: ['edge_left'], action: 'spin_right' }),
  ]);
  const [def, setDef] = useState<OpponentAction>('spin_left');

  // Hardware — the full spec the opponent is built on (used in sim).
  const [hwSpec, setHwSpec] = useState<HardwareSpec | null>(null);
  // Whether the user has edited the hardware since it was last seeded (so a
  // preset click can confirm before discarding edits).
  const [hwDirty, setHwDirty] = useState(false);
  const [geom, setGeom] = useState<Geometry | null>(null);
  const [geomLoading, setGeomLoading] = useState(false);
  const [geomError, setGeomError] = useState<string | null>(null);
  const geomRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Optional "seed from a saved robot" picker.
  const [robots, setRobots] = useState<RobotSummary[]>([]);

  // Stage-1 hardware validation (debounced) — surfaces the backend's mini-sumo
  // class violations (and any URDF error) BEFORE the user reaches save, so an
  // over-limit chassis reads e.g. "mass 800 g exceeds the mini-sumo limit…".
  const [hwErrors, setHwErrors] = useState<string[]>([]);
  const hwValidateRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Validation (debounced)
  const [validOk, setValidOk] = useState<boolean | null>(null);
  const [validErrors, setValidErrors] = useState<string[]>([]);
  const validateRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Save
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null);
  const [savedNotes, setSavedNotes] = useState<string | null>(null);

  // Test battle
  const [models, setModels] = useState<ModelCard[]>([]);
  const [testModelId, setTestModelId] = useState('');
  const [testingId, setTestingId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<
    { oppName: string; result: BattleResult; oppSpec: HardwareSpec | null } | null
  >(null);
  const [testError, setTestError] = useState<string | null>(null);

  const dsl = useMemo(() => compileDsl(rules, def), [rules, def]);

  // The behavior object Stage 2 currently describes (zoo selection or the
  // compiled DSL). Null when a zoo source is chosen but no id is selected yet.
  const behavior = useMemo<OpponentBehavior | null>(() => {
    if (behaviorSource === 'zoo') {
      return zooId ? { kind: 'zoo', zoo_id: zooId } : null;
    }
    return { kind: 'dsl', dsl };
  }, [behaviorSource, zooId, dsl]);

  // ---- Loaders -------------------------------------------------------------
  const reloadRoster = useCallback(async () => {
    try {
      const list = await api.listOpponents();
      setRoster(list);
      setRosterError(null);
      // Fetch rule counts lazily for DSL opponents only (list summaries omit
      // the DSL; zoo opponents have no rules so we leave them undefined).
      const counts: Record<string, number> = {};
      await Promise.all(
        list.map(async (o) => {
          if (o.behavior?.kind === 'zoo') return;
          try {
            const full = await api.getOpponent(o.id);
            if (full.behavior?.kind === 'dsl') {
              counts[o.id] = full.behavior.dsl.rules.length;
            } else if (full.behavior_dsl) {
              counts[o.id] = full.behavior_dsl.rules.length;
            }
          } catch {
            /* leave undefined */
          }
        }),
      );
      setRuleCounts(counts);
    } catch (e) {
      setRosterError(e instanceof ApiError ? e.message : 'Failed to load opponents.');
    }
  }, []);

  useEffect(() => {
    reloadRoster();
    api
      .hardwareDefault()
      .then((s) => {
        setHwSpec(s);
        setHwDirty(false);
      })
      .catch(() => {});
    api
      .listRobots()
      .then(setRobots)
      .catch(() => {});
    api
      .trainOpponents()
      .then(({ opponents }) => {
        setZooList(opponents);
        // Default the zoo dropdown to the first id so a "built-in" save is
        // ready without an extra click.
        setZooId((cur) => cur || opponents[0]?.id || '');
      })
      .catch(() => {});
    api
      .models()
      .then((ms) => {
        setModels(ms);
        const robust = ms.find((m) => m.id === 'ppo_robust_best');
        const canonical = ms.find((m) => m.obs_dim === 21 && m.action_dim === 9);
        setTestModelId((cur) => cur || (robust ?? canonical ?? ms[0])?.id || '');
      })
      .catch(() => {});
  }, [reloadRoster]);

  // ---- Debounced 3D geometry for the live preview --------------------------
  useEffect(() => {
    if (!hwSpec) return;
    if (geomRef.current) clearTimeout(geomRef.current);
    geomRef.current = setTimeout(async () => {
      setGeomLoading(true);
      try {
        const g = await api.geometry(hwSpec);
        setGeom(g);
        setGeomError(null);
      } catch (e) {
        setGeomError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setGeomLoading(false);
      }
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      if (geomRef.current) clearTimeout(geomRef.current);
    };
  }, [hwSpec]);

  // ---- Debounced hardware validate (mini-sumo class + URDF) ----------------
  // Reuses /api/hardware/validate (same endpoint the Hardware page uses); its
  // `errors[]` carries the mini-sumo violations the backend would reject on save.
  useEffect(() => {
    if (!hwSpec) return;
    if (hwValidateRef.current) clearTimeout(hwValidateRef.current);
    hwValidateRef.current = setTimeout(async () => {
      try {
        const v = await api.validate(hwSpec);
        setHwErrors(v.errors);
      } catch {
        // Non-fatal: leave any prior errors; the save path still gates.
      }
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      if (hwValidateRef.current) clearTimeout(hwValidateRef.current);
    };
  }, [hwSpec]);

  // Seed the hardware editor from a saved robot (optional convenience).
  async function loadFromRobot(id: string) {
    if (!id) return;
    try {
      const rec = await api.getRobot(id);
      setHwSpec(rec.hardware_spec);
      setHwDirty(false);
    } catch (e) {
      setSaveMsg({
        kind: 'err',
        text: e instanceof ApiError ? e.message : 'Could not load that robot.',
      });
    }
  }

  // Seed the hardware editor from a preset chassis. Confirm first if the user
  // has already edited the current spec (so a stray click can't discard work).
  function seedFromPreset(spec: HardwareSpec) {
    if (
      hwDirty &&
      !window.confirm(
        'Replace the current hardware with this preset? Your edits will be lost.',
      )
    ) {
      return;
    }
    setHwSpec(spec);
    setHwDirty(false);
  }

  // ---- Debounced validate --------------------------------------------------
  // Only the DSL source needs a live backend round-trip; a zoo behavior is
  // valid as soon as an id is selected (the dropdown only offers known ids).
  useEffect(() => {
    if (behaviorSource !== 'dsl') {
      if (validateRef.current) clearTimeout(validateRef.current);
      setValidOk(zooId ? true : null);
      setValidErrors([]);
      return;
    }
    if (validateRef.current) clearTimeout(validateRef.current);
    setValidOk(null);
    validateRef.current = setTimeout(async () => {
      try {
        const res = await api.validateOpponentDsl(dsl);
        setValidOk(res.ok);
        setValidErrors(res.errors);
      } catch {
        setValidOk(false);
        setValidErrors(['Validation request failed — is the backend up?']);
      }
    }, VALIDATE_DEBOUNCE_MS);
    return () => {
      if (validateRef.current) clearTimeout(validateRef.current);
    };
  }, [dsl, behaviorSource, zooId]);

  // ---- Rule mutations ------------------------------------------------------
  function patchRule(key: string, patch: Partial<RuleDraft>) {
    setRules((rs) => rs.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  }
  function addRule() {
    setRules((rs) => [...rs, newRule()]);
  }
  function removeRule(key: string) {
    setRules((rs) => rs.filter((r) => r.key !== key));
  }
  function moveRule(key: string, dir: -1 | 1) {
    setRules((rs) => {
      const i = rs.findIndex((r) => r.key === key);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= rs.length) return rs;
      const next = [...rs];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  }
  function togglePred(key: string, pred: OpponentPredicate) {
    setRules((rs) =>
      rs.map((r) => {
        if (r.key !== key) return r;
        const has = r.preds.includes(pred);
        const preds = has
          ? r.preds.filter((p) => p !== pred)
          : [...r.preds, pred];
        return { ...r, preds };
      }),
    );
  }

  // ---- Save ----------------------------------------------------------------
  async function save() {
    if (!hwSpec) {
      setSaveMsg({ kind: 'err', text: 'Hardware spec not loaded yet.' });
      return;
    }
    if (!name.trim()) {
      setSaveMsg({ kind: 'err', text: 'Give the opponent a name first.' });
      return;
    }
    if (!behavior) {
      setSaveMsg({ kind: 'err', text: 'Pick a built-in behavior first.' });
      return;
    }
    setSaving(true);
    setSaveMsg(null);
    setSavedNotes(null);
    try {
      const rec = await api.createOpponent({
        name: name.trim(),
        hardware_spec: hwSpec,
        behavior,
      });
      setSaveMsg({ kind: 'ok', text: `Saved “${rec.name}” (${rec.id}).` });
      setSavedNotes(rec.notes ?? null);
      setName('');
      setStage('hardware');
      await reloadRoster();
    } catch (e) {
      setSaveMsg({
        kind: 'err',
        text: e instanceof ApiError ? e.message : 'Save failed.',
      });
    } finally {
      setSaving(false);
    }
  }

  // ---- Delete --------------------------------------------------------------
  async function del(o: CustomOpponentSummary) {
    if (!window.confirm(`Delete custom opponent “${o.name}”? This can't be undone.`)) {
      return;
    }
    try {
      await api.deleteOpponent(o.id);
      if (testResult?.oppName === o.name) setTestResult(null);
      await reloadRoster();
    } catch (e) {
      setRosterError(e instanceof ApiError ? e.message : 'Delete failed.');
    }
  }

  // ---- Test (tiny: 1 round) ------------------------------------------------
  async function test(o: CustomOpponentSummary) {
    if (!testModelId) {
      setTestError('Pick a model to test against.');
      return;
    }
    setTestingId(o.id);
    setTestError(null);
    setTestResult(null);
    try {
      // Fetch the full record so the replay can render the opponent on its OWN
      // built hardware (the list summary omits the spec). Non-fatal if it fails.
      const [result, full] = await Promise.all([
        api.runBattle({
          a_model_id: testModelId,
          b_opponent_id: o.id,
          rounds: 1,
        }),
        api.getOpponent(o.id).catch(() => null),
      ]);
      setTestResult({
        oppName: o.name,
        result,
        oppSpec: full?.hardware_spec ?? null,
      });
    } catch (e) {
      setTestError(e instanceof ApiError ? e.message : 'Test battle failed.');
    } finally {
      setTestingId(null);
    }
  }

  const saveDisabled =
    saving || validOk !== true || !name.trim() || !hwSpec || !behavior;

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      {/* ============================ CREATE ============================ */}
      <div className="flex flex-col gap-5">
        {/* Stage indicator */}
        <Reveal index={0}>
          <Panel title="Forge Opponent" live bodyClassName="p-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <StageChip n={1} label="HARDWARE" active={stage === 'hardware'} done={stage === 'logic'} />
                <span className="num text-fg-2" style={{ fontSize: 12 }}>
                  →
                </span>
                <StageChip n={2} label="LOGIC" active={stage === 'logic'} done={false} />
              </div>
              <span className="num text-fg-2" style={{ fontSize: 10 }}>
                {stage === 'hardware'
                  ? 'Build the chassis the opponent fights on.'
                  : 'Now define how it behaves, then name + save.'}
              </span>
            </div>
          </Panel>
        </Reveal>

        {stage === 'hardware' ? (
          /* ---------------- STAGE 1 · HARDWARE ---------------- */
          <div className="grid grid-cols-1 gap-5 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="flex flex-col gap-5">
              {/* Seed-from-preset + seed-from-robot pickers */}
              <Reveal index={1}>
                <Panel title="Seed Hardware" live bodyClassName="flex flex-col gap-3 p-4">
                  {/* Preset chips — drop in a ready-made chassis. */}
                  <HardwarePresetPicker onPick={(spec) => seedFromPreset(spec)} />

                  <div className="h-px w-full" style={{ background: 'var(--line)' }} />

                  <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
                    LOAD FROM A SAVED ROBOT · OPTIONAL
                    <Info topic="opp_hardware" />
                  </span>
                  <select
                    className="ctl num"
                    defaultValue=""
                    onChange={(e) => {
                      loadFromRobot(e.target.value);
                      e.target.value = '';
                    }}
                    style={{ fontSize: 12 }}
                  >
                    <option value="">— start from default chassis —</option>
                    {robots.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.name} · {r.id}
                      </option>
                    ))}
                  </select>
                  <span className="num text-fg-2" style={{ fontSize: 10, lineHeight: 1.5 }}>
                    Drop in a preset, copy a saved robot's full spec, or edit the
                    default chassis directly. The opponent fights on exactly this
                    hardware.
                  </span>
                </Panel>
              </Reveal>

              {hwSpec ? (
                <HardwareForm
                  spec={hwSpec}
                  setSpec={(updater) =>
                    setHwSpec((prev) => {
                      if (!prev) return prev;
                      setHwDirty(true);
                      return updater(prev);
                    })
                  }
                  startIndex={2}
                />
              ) : (
                <Panel title="Hardware" live>
                  <span className="num text-fg-2" style={{ fontSize: 11 }}>
                    Loading default chassis…
                  </span>
                </Panel>
              )}

              <Reveal index={8}>
                <div className="flex flex-col gap-2">
                  {/* Mini-sumo / URDF validation — surfaced before BEHAVIOR so an
                      over-limit chassis is caught here, not only at save. */}
                  {hwErrors.length > 0 && (
                    <div
                      className="flex flex-col gap-1 rounded border px-3 py-2"
                      style={{
                        borderColor: 'var(--loss)',
                        background: 'rgba(255,84,112,.08)',
                      }}
                    >
                      <span className="micro inline-flex items-center gap-1.5" style={{ fontSize: 10, color: 'var(--loss)' }}>
                        NOT MINI-SUMO LEGAL
                        <Info topic="mini_sumo" />
                      </span>
                      {hwErrors.map((e, i) => (
                        <span key={i} className="num text-fg-1" style={{ fontSize: 10 }}>
                          • {e}
                        </span>
                      ))}
                    </div>
                  )}
                  <button
                    className="btn btn-primary"
                    onClick={() => setStage('logic')}
                    disabled={!hwSpec || hwErrors.length > 0}
                    style={{ height: 40, fontSize: 14 }}
                  >
                    NEXT: BEHAVIOR →
                  </button>
                </div>
              </Reveal>
            </div>

            {/* Live preview of the opponent being built */}
            <Reveal index={1} className="2xl:sticky 2xl:top-5 2xl:self-start">
              <RobotPreview
                geom={geom}
                spec={hwSpec}
                loading={geomLoading}
                error={geomError}
              />
            </Reveal>
          </div>
        ) : (
          /* ---------------- STAGE 2 · BEHAVIOR ---------------- */
          <Reveal index={1}>
            <Panel title="Behavior" live ticks bodyClassName="flex flex-col gap-5 p-5">
              {/* Back */}
              <button
                className="btn btn-ghost self-start"
                onClick={() => setStage('hardware')}
                style={{ height: 30, fontSize: 11 }}
              >
                ← BACK TO HARDWARE
              </button>

              {/* Name */}
              <label className="flex flex-col gap-1">
                <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                  OPPONENT NAME
                </span>
                <input
                  className="ctl num"
                  value={name}
                  placeholder="e.g. Heavy Dodger"
                  onChange={(e) => setName(e.target.value)}
                  style={{ fontSize: 13 }}
                />
              </label>

              {/* Behavior source toggle */}
              <div className="flex flex-col gap-2">
                <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
                  BEHAVIOR SOURCE
                  <Info topic="opp_behavior_source" />
                </span>
                <Seg
                  value={behaviorSource}
                  onChange={setBehaviorSource}
                  options={[
                    { value: 'zoo', label: 'Built-in' },
                    { value: 'dsl', label: 'Custom Rules' },
                  ]}
                />
              </div>

              {behaviorSource === 'zoo' ? (
                /* ---- Built-in zoo controller ---- */
                <div className="flex flex-col gap-2">
                  <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
                    BUILT-IN CONTROLLER
                    <Info topic="opp_zoo" />
                  </span>
                  <div className="flex items-center gap-1.5">
                    <select
                      className="ctl num"
                      value={zooId}
                      onChange={(e) => setZooId(e.target.value)}
                      style={{ fontSize: 12, flex: 1 }}
                    >
                      {zooList.length === 0 && <option value="">— loading… —</option>}
                      {zooList.map((o) => (
                        <option key={o.id} value={o.id}>
                          {o.id}
                          {o.held_out ? ' · eval-only' : ''}
                        </option>
                      ))}
                    </select>
                    {/* ⓘ describes whichever controller is selected. */}
                    {zooInfoTopic(zooId) && <Info topic={zooInfoTopic(zooId)!} />}
                  </div>
                  <span className="num text-fg-2" style={{ fontSize: 10, lineHeight: 1.5 }}>
                    One of our scripted zoo bots, dropped onto the hardware you
                    built. Pair e.g. dodger with a heavy chassis for a "Heavy
                    Dodger".
                  </span>
                </div>
              ) : (
                /* ---- Custom rule DSL ---- */
                <>
                  {/* Rules */}
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
                        BEHAVIOR RULES · CHECKED TOP→BOTTOM
                        <Info topic="opp_rules" />
                      </span>
                      <button className="btn btn-secondary" onClick={addRule} style={{ height: 28, fontSize: 11 }}>
                        + RULE
                      </button>
                    </div>

                    {rules.length === 0 && (
                      <span className="num" style={{ fontSize: 11, color: 'var(--warn)' }}>
                        No rules yet — add at least one (the validator requires it).
                      </span>
                    )}

                    {rules.map((r, i) => (
                      <RuleEditor
                        key={r.key}
                        rule={r}
                        index={i}
                        count={rules.length}
                        onPatch={(patch) => patchRule(r.key, patch)}
                        onTogglePred={(p) => togglePred(r.key, p)}
                        onRemove={() => removeRule(r.key)}
                        onMove={(dir) => moveRule(r.key, dir)}
                      />
                    ))}
                  </div>

                  {/* Default */}
                  <label className="flex flex-col gap-1">
                    <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
                      DEFAULT ACTION · WHEN NO RULE MATCHES
                      <Info topic="opp_default" />
                    </span>
                    <ActionSelect value={def} onChange={setDef} />
                  </label>
                </>
              )}

              {/* Live validation */}
              <ValidateBanner ok={validOk} errors={validErrors} />

              {/* Save */}
              <div className="flex flex-col gap-2">
                <button
                  className="btn btn-primary"
                  disabled={saveDisabled}
                  onClick={save}
                  style={{ height: 40, fontSize: 15 }}
                >
                  {saving ? 'SAVING…' : '⛭ SAVE OPPONENT'}
                </button>
                {saveMsg && (
                  <span
                    className="num"
                    style={{
                      fontSize: 11,
                      color: saveMsg.kind === 'ok' ? 'var(--win)' : 'var(--loss)',
                    }}
                  >
                    {saveMsg.text}
                  </span>
                )}
                {savedNotes && (
                  <span className="num text-fg-2" style={{ fontSize: 10, lineHeight: 1.5 }}>
                    ⓘ {savedNotes}
                  </span>
                )}
              </div>
            </Panel>
          </Reveal>
        )}
      </div>

      {/* ============================ ROSTER + TEST ============================ */}
      <div className="flex flex-col gap-5">
        <Reveal index={1}>
          <Panel
            title="Custom Roster"
            live
            ticks
            bodyClassName="flex flex-col gap-3 p-5"
            right={
              <span className="num text-fg-2" style={{ fontSize: 10 }}>
                {roster.length} SAVED
              </span>
            }
          >
            {/* Test-against model picker */}
            <label className="flex flex-col gap-1">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                TEST AGAINST · SIDE-A MODEL
              </span>
              <select
                className="ctl num"
                value={testModelId}
                onChange={(e) => setTestModelId(e.target.value)}
                style={{ fontSize: 12 }}
              >
                {models.length === 0 && <option value="">— no models —</option>}
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} · {m.algo} · {m.obs_dim}obs/{m.action_dim}act
                  </option>
                ))}
              </select>
              <span className="num text-fg-2" style={{ fontSize: 10 }}>
                TEST runs ONE round of real physics vs the selected model.
              </span>
            </label>

            {rosterError && (
              <span className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
                {rosterError}
              </span>
            )}

            {roster.length === 0 ? (
              <div className="flex flex-col items-center gap-2 py-6">
                <StatusPill status="idle" label="NO CUSTOM OPPONENTS" />
                <p className="num text-center text-fg-2" style={{ fontSize: 11, maxWidth: 320 }}>
                  Build a behavior on the left and SAVE — it appears here and in the
                  Arena side-B picker.
                </p>
              </div>
            ) : (
              roster.map((o) => (
                <OpponentCardRow
                  key={o.id}
                  opp={o}
                  ruleCount={ruleCounts[o.id] ?? null}
                  onDelete={() => del(o)}
                  onTest={() => test(o)}
                  testing={testingId === o.id}
                  testDisabled={!testModelId || testingId !== null}
                />
              ))
            )}

            {testError && (
              <span className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
                {testError}
              </span>
            )}
          </Panel>
        </Reveal>

        {/* Test replay */}
        {testResult && (
          <Reveal index={2}>
            <Panel
              title={`Test Battle · ${testResult.oppName}`}
              live
              ticks
              bodyClassName="flex flex-col gap-3 p-5"
            >
              <TestVerdict result={testResult.result} />
              <div className="min-h-[360px]">
                <TrajectoryPlayer
                  traj={testResult.result.trajectory ?? null}
                  opponentSpec={testResult.oppSpec}
                  label={`${testModelId} vs ${testResult.oppName}`}
                />
              </div>
            </Panel>
          </Reveal>
        )}
      </div>
    </div>
  );
}

// ===========================================================================
// Sub-components
// ===========================================================================

const ACTION_INFO: Record<OpponentAction, string> = {
  forward: 'act_forward',
  reverse: 'act_reverse',
  spin_left: 'act_spin_left',
  spin_right: 'act_spin_right',
  arc_left: 'act_arc_left',
  arc_right: 'act_arc_right',
  stop: 'act_stop',
};

// Segmented toggle (mirrors Train/Arena's Seg primitive).
function Seg<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex overflow-hidden rounded border" style={{ borderColor: 'var(--line)' }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            className="font-display uppercase"
            style={{
              flex: 1,
              fontSize: 12,
              letterSpacing: '.06em',
              padding: '7px 10px',
              border: 'none',
              cursor: 'pointer',
              color: active ? 'var(--bg-0)' : 'var(--fg-1)',
              background: active ? 'var(--accent)' : 'var(--bg-2)',
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function ActionSelect({
  value,
  onChange,
}: {
  value: OpponentAction;
  onChange: (a: OpponentAction) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <select
        className="ctl num"
        value={value}
        onChange={(e) => onChange(e.target.value as OpponentAction)}
        style={{ fontSize: 12, flex: 1 }}
      >
        {ACTIONS.map((a) => (
          <option key={a} value={a}>
            {ACTION_LABEL[a]} ({a})
          </option>
        ))}
      </select>
      {/* ⓘ explains whatever action is currently selected. */}
      <Info topic={ACTION_INFO[value]} />
    </div>
  );
}

function RuleEditor({
  rule,
  index,
  count,
  onPatch,
  onTogglePred,
  onRemove,
  onMove,
}: {
  rule: RuleDraft;
  index: number;
  count: number;
  onPatch: (patch: Partial<RuleDraft>) => void;
  onTogglePred: (p: OpponentPredicate) => void;
  onRemove: () => void;
  onMove: (dir: -1 | 1) => void;
}) {
  return (
    <div
      className="flex flex-col gap-3 rounded border p-3"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      {/* Header row: index + reorder/remove */}
      <div className="flex items-center justify-between">
        <span className="micro" style={{ fontSize: 10, color: 'var(--accent)' }}>
          RULE {index + 1}
        </span>
        <div className="flex items-center gap-1">
          <IconBtn label="↑" title="Move up" disabled={index === 0} onClick={() => onMove(-1)} />
          <IconBtn label="↓" title="Move down" disabled={index === count - 1} onClick={() => onMove(1)} />
          <IconBtn label="✕" title="Remove rule" tone="loss" onClick={onRemove} />
        </div>
      </div>

      {/* WHEN */}
      <div className="flex flex-col gap-2">
        <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 9 }}>
          WHEN
          <Info topic="opp_when" />
        </span>

        {/* Timer toggle */}
        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={rule.useTimer}
            onChange={(e) => onPatch({ useTimer: e.target.checked })}
            style={{ accentColor: 'var(--accent)' }}
          />
          <span className="num inline-flex items-center gap-1.5" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
            Use a TIMER instead of sensors
            <Info topic="opp_timer" />
          </span>
        </label>

        {rule.useTimer ? (
          <label className="flex items-center gap-2">
            <span className="num text-fg-2" style={{ fontSize: 11 }}>
              Every
            </span>
            <input
              type="number"
              className="ctl num"
              min={1}
              step={1}
              value={rule.timerEvery}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (Number.isFinite(v)) onPatch({ timerEvery: v });
              }}
              style={{ width: 90, height: 30, fontSize: 12 }}
            />
            <span className="num text-fg-2" style={{ fontSize: 11 }}>
              ticks
            </span>
          </label>
        ) : (
          <>
            {/* Predicate chips */}
            <div className="flex flex-wrap gap-1.5">
              {PREDICATES.map((p) => {
                const on = rule.preds.includes(p);
                return (
                  <button
                    key={p}
                    type="button"
                    onClick={() => onTogglePred(p)}
                    className="num inline-flex items-center gap-1"
                    style={{
                      fontSize: 10,
                      padding: '3px 7px',
                      borderRadius: 'var(--radius)',
                      border: `1px solid ${on ? 'var(--cyan)' : 'var(--line)'}`,
                      background: on ? 'var(--cyan-glow)' : 'var(--bg-1)',
                      color: on ? 'var(--cyan)' : 'var(--fg-1)',
                      cursor: 'pointer',
                    }}
                  >
                    {PREDICATE_LABEL[p]}
                    <Info topic={PREDICATE_INFO[p]} />
                  </button>
                );
              })}
            </div>

            {/* Combine + NOT */}
            <div className="flex flex-wrap items-center gap-3">
              {rule.preds.length > 1 && (
                <label className="flex items-center gap-1.5">
                  <span className="micro inline-flex items-center gap-1 text-fg-2" style={{ fontSize: 9 }}>
                    COMBINE
                    <Info topic="opp_combine" />
                  </span>
                  <select
                    className="ctl num"
                    value={rule.combine}
                    onChange={(e) => onPatch({ combine: e.target.value as Combine })}
                    style={{ width: 90, height: 28, fontSize: 11 }}
                  >
                    <option value="all">ALL of</option>
                    <option value="any">ANY of</option>
                  </select>
                </label>
              )}
              <label className="flex cursor-pointer items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={rule.not}
                  onChange={(e) => onPatch({ not: e.target.checked })}
                  style={{ accentColor: 'var(--accent)' }}
                />
                <span className="num inline-flex items-center gap-1" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
                  NOT (invert)
                  <Info topic="opp_not" />
                </span>
              </label>
            </div>
            {rule.preds.length === 0 && (
              <span className="num" style={{ fontSize: 10, color: 'var(--warn)' }}>
                Pick at least one predicate (or switch to a timer).
              </span>
            )}
          </>
        )}
      </div>

      {/* DO */}
      <div className="flex flex-col gap-1">
        <span className="micro text-fg-2" style={{ fontSize: 9 }}>
          DO
        </span>
        <ActionSelect value={rule.action} onChange={(a) => onPatch({ action: a })} />
      </div>
    </div>
  );
}

function IconBtn({
  label,
  title,
  onClick,
  disabled,
  tone,
}: {
  label: string;
  title: string;
  onClick: () => void;
  disabled?: boolean;
  tone?: 'loss';
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      disabled={disabled}
      className="num inline-flex items-center justify-center"
      style={{
        width: 22,
        height: 22,
        fontSize: 11,
        borderRadius: 'var(--radius)',
        border: '1px solid var(--line)',
        background: 'var(--bg-1)',
        color: disabled ? 'var(--fg-2)' : tone === 'loss' ? 'var(--loss)' : 'var(--fg-1)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
      }}
    >
      {label}
    </button>
  );
}

function ValidateBanner({
  ok,
  errors,
}: {
  ok: boolean | null;
  errors: string[];
}) {
  const color =
    ok === true ? 'var(--win)' : ok === false ? 'var(--loss)' : 'var(--idle)';
  const label =
    ok === true ? 'DSL VALID' : ok === false ? 'DSL INVALID' : 'CHECKING…';
  return (
    <div
      className="flex flex-col gap-1.5 rounded border px-3 py-2"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-2 w-2 rounded-full"
          style={{ background: color, boxShadow: `0 0 8px ${color}` }}
        />
        <span className="micro" style={{ fontSize: 10, color, letterSpacing: '.1em' }}>
          {label}
        </span>
      </div>
      {ok === false &&
        errors.map((e, i) => (
          <span key={i} className="num" style={{ fontSize: 10, color: 'var(--loss)' }}>
            • {e}
          </span>
        ))}
    </div>
  );
}

function StageChip({
  n,
  label,
  active,
  done,
}: {
  n: number;
  label: string;
  active: boolean;
  done: boolean;
}) {
  const color = active ? 'var(--accent)' : done ? 'var(--cyan)' : 'var(--fg-2)';
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="num inline-flex items-center justify-center"
        style={{
          width: 18,
          height: 18,
          fontSize: 10,
          borderRadius: '50%',
          border: `1px solid ${color}`,
          color,
          background: active ? 'var(--accent-glow)' : 'transparent',
        }}
      >
        {done ? '✓' : n}
      </span>
      <span
        className="micro"
        style={{ fontSize: 11, letterSpacing: '.1em', color }}
      >
        {label}
      </span>
    </span>
  );
}

function OpponentCardRow({
  opp,
  ruleCount,
  onDelete,
  onTest,
  testing,
  testDisabled,
}: {
  opp: CustomOpponentSummary;
  ruleCount: number | null;
  onDelete: () => void;
  onTest: () => void;
  testing: boolean;
  testDisabled: boolean;
}) {
  return (
    <div
      className="flex items-center justify-between gap-3 rounded border px-3 py-2.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="font-display uppercase" style={{ fontSize: 14, color: 'var(--cyan)', letterSpacing: '.03em' }}>
          {opp.name}
        </span>
        {/* Behavior summary chip + (for DSL) rule count. */}
        <span className="num inline-flex flex-wrap items-center gap-1.5" style={{ fontSize: 10 }}>
          {opp.behavior_summary && (
            <span
              style={{
                color: opp.behavior?.kind === 'zoo' ? 'var(--accent)' : 'var(--cyan)',
                border: '1px solid var(--line-2)',
                borderRadius: 'var(--radius)',
                padding: '1px 6px',
                letterSpacing: '.02em',
              }}
            >
              {opp.behavior_summary}
            </span>
          )}
          {opp.behavior?.kind !== 'zoo' && (
            <span className="text-fg-2">{ruleCountLabel(ruleCount)}</span>
          )}
          <span className="text-fg-2">· {fmtDate(opp.created_at)}</span>
        </span>
        <span className="num text-fg-2" style={{ fontSize: 9 }}>
          {opp.id}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <button
          className="btn btn-secondary"
          onClick={onTest}
          disabled={testDisabled}
          style={{ height: 30, fontSize: 11 }}
        >
          {testing ? 'TESTING…' : '⚔ TEST'}
        </button>
        <button
          className="btn btn-ghost"
          onClick={onDelete}
          style={{ height: 30, fontSize: 11, color: 'var(--loss)' }}
        >
          DELETE
        </button>
      </div>
    </div>
  );
}

function TestVerdict({ result }: { result: BattleResult }) {
  // Opponent tests always run a single 1-round battle, so `stats` is present;
  // fall back defensively in case a future caller passes a gauntlet result.
  const s = result.stats ?? {
    rounds: 0,
    a_wins: 0,
    b_wins: 0,
    draws: 0,
    timeouts: 0,
    a_self_out: 0,
    b_self_out: 0,
  };
  const verdict =
    s.a_wins > s.b_wins
      ? { label: 'MODEL WINS', color: 'var(--accent)' }
      : s.b_wins > s.a_wins
        ? { label: 'OPPONENT WINS', color: 'var(--cyan)' }
        : { label: 'PUSH / TIMEOUT', color: 'var(--warn)' };
  return (
    <div className="flex items-center justify-between">
      <div className="flex flex-col">
        <span className="micro text-fg-2" style={{ fontSize: 10 }}>
          VERDICT · 1 ROUND
        </span>
        <span
          className="font-display uppercase"
          style={{ fontSize: 20, color: verdict.color, letterSpacing: '.04em' }}
        >
          {verdict.label}
        </span>
      </div>
      <div className="num text-fg-2" style={{ fontSize: 11, textAlign: 'right' }}>
        <span style={{ color: 'var(--accent)' }}>A wins {s.a_wins}</span>
        {' · '}
        <span style={{ color: 'var(--cyan)' }}>B wins {s.b_wins}</span>
        <br />
        self-out A {s.a_self_out} · B {s.b_self_out}
      </div>
    </div>
  );
}
