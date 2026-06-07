# SumoForge — Design System ("Dohyo Mission Console")

Source of truth for the lite frontend. Aesthetic: an **industrial robotics
instrument panel / mission console** — precise, dense, technical. Deep cool-black
base, a **forge-orange** combat accent, **cyan** telemetry, angular technical
type, faint engineering-grid texture, and a HUD-framed 3D viewport. NOT a generic
SaaS dashboard. No purple-on-white, no Inter/Roboto/Space Grotesk.

## Typography (Google Fonts)
- **Display / UI labels:** `Oxanium` (angular, mechanical) — headings, nav, buttons, wordmark. Uppercase + tracking for labels.
- **Body / data / numbers:** `IBM Plex Mono` — telemetry, tables, metrics, code/DSL.
- Scale: 11px micro-labels (uppercase, letter-spacing .12em), 13px body, 15px controls, 20/28/40px headings. Numbers always mono + tabular-nums.

## Color tokens (CSS variables — put on :root)
```css
:root{
  /* base (deep, cool) */
  --bg-0:#080b0f;   /* app background (deepest) */
  --bg-1:#0f151b;   /* panels */
  --bg-2:#172029;   /* raised / table stripes / inputs */
  --bg-3:#1f2a35;   /* hover */
  --line:#26323d;   /* hairline borders + grid */
  --line-2:#33424f; /* stronger divider */
  /* text */
  --fg-0:#e8eef4;   /* primary */
  --fg-1:#9babb9;   /* muted */
  --fg-2:#5f7180;   /* faint / placeholders */
  /* accent — forge / combat */
  --accent:#ff7a18;       /* primary action, agent robot, active nav */
  --accent-hot:#ff4d2e;   /* emphasis / danger-adjacent */
  --accent-dim:#b8560f;
  --accent-glow:rgba(255,122,24,.35);
  /* telemetry — cyan */
  --cyan:#2ad4ff;
  --cyan-dim:#1789a8;
  --cyan-glow:rgba(42,212,255,.30);
  /* status */
  --win:#46d39a; --loss:#ff5470; --warn:#ffcc33; --idle:#5f7180;
  /* fx */
  --grid:rgba(255,255,255,.025);
  --radius:6px;            /* tight, technical corners */
  --shadow:0 10px 30px -12px rgba(0,0,0,.7);
}
```
Robot colors in 3D + UI: **agent = `--accent` (forge orange)**, **opponent = `--cyan`**.

## Tailwind theme (tailwind.config — extend.colors map to the vars)
```js
colors:{
  bg:{0:'var(--bg-0)',1:'var(--bg-1)',2:'var(--bg-2)',3:'var(--bg-3)'},
  line:{DEFAULT:'var(--line)',2:'var(--line-2)'},
  fg:{0:'var(--fg-0)',1:'var(--fg-1)',2:'var(--fg-2)'},
  accent:{DEFAULT:'var(--accent)',hot:'var(--accent-hot)',dim:'var(--accent-dim)'},
  cyan:{DEFAULT:'var(--cyan)',dim:'var(--cyan-dim)'},
  win:'var(--win)',loss:'var(--loss)',warn:'var(--warn)',
},
fontFamily:{ display:['Oxanium','sans-serif'], mono:['"IBM Plex Mono"','monospace'] },
borderRadius:{ DEFAULT:'var(--radius)' },
```

## Layout language — left-nav app shell
- **Left rail** (fixed, ~228px, `--bg-1`, right border `--line`): wordmark **SUMOFORGE** at top (Oxanium 700, the "FORGE" in `--accent`), then nav items (icon + uppercase label): `Hardware`, `Train`, `Arena`, `Opponents`, `Models`. Active item: left 2px `--accent` bar + faint `--accent-glow` wash + brightened label. A small "engine status" footer chip (sumo env / backend connected = cyan dot).
- **Top bar** (in content area): section title (Oxanium uppercase) + breadcrumb, right-aligned status chips (backend `●`, running-job pill).
- **Panels:** `--bg-1`, 1px `--line` border, `--radius`, optional 1px top accent line when "active/live". Header row = uppercase micro-label + thin divider. Use **corner ticks** (small L-shaped brackets in `--line-2`) on hero panels (viewport, builder preview) for the instrument look.
- **Background:** `--bg-0` with a faint engineering grid (`--grid` lines ~32px) + a very low-opacity noise/grain overlay. Subtle radial vignette toward edges.
- Density: compact controls (32–34px), 12–16px gaps, mono numeric readouts everywhere.

## Components
- **Buttons:** primary = solid `--accent` on `--bg-0` text, slight glow on hover; secondary = `--bg-2` + `--line` border; ghost = text only. Uppercase Oxanium, letter-spacing .06em. Sharp corners.
- **Inputs/sliders:** `--bg-2`, `--line` border, focus ring `--cyan`; sliders with a mono value bubble; "RECOMMENDED" defaults shown as a faint ghost value + a reset chevron.
- **Status pills:** dot + uppercase micro-label; win=`--win`, self-out/loss=`--loss`, training=`--accent` pulsing, idle=`--idle`.
- **Tables (Models/metrics):** mono, zebra via `--bg-2`, win-rate as a thin horizontal bar in `--accent`/`--cyan`.
- **Cards (model/opponent):** header strip + key stats grid (WR, self-out%, net size, trained-on-hardware), signature hash in `--fg-2` mono.

## 3D replay viewport (the centerpiece — "DOHYO CAM")
- Dark stage (`--bg-0`→ radial slightly lighter center), faint grid floor, soft vignette, a subtle scanline overlay + low grain. **Corner brackets** framing the canvas.
- HUD overlay (mono, non-interactive): top-left step/frame + mult; top-right outcome chip (WIN/SELF-OUT/PUSH/TIMEOUT in status colors); a thin perimeter ring for the dohyo edge.
- Robots from `/api/hardware/geometry` primitives: agent `--accent`, opponent `--cyan`, with a faint emissive rim. Soft key light + ambient; contact shadow on the floor.
- **Transport bar** styled like a console: play/pause, scrub (frame timeline), speed (0.5/1/2×), step counter. Cyan playhead.
- Powering-on micro-animation when a replay loads (quick flicker + grid fade-in).

## Motion (Motion/`framer-motion` for React)
- Page load: staggered panel reveal (translateY+fade, 40–60ms stagger).
- Active nav glow; value count-up for metrics; training charts stream in.
- Restraint elsewhere — this is an instrument, not a toy. High-impact moments only.

## App sections (routes)
`/hardware` (Hardware Builder + live 3D preview + validate), `/train` (config + recommended HPs + live dashboard + replay), `/arena` (battle setup + replay + stats), `/opponents` (rule-DSL builder + hardware), `/models` (registry table + cards). Lite build: single user, talks to FastAPI at `http://127.0.0.1:8000`.
