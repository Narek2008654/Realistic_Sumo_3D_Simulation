// Plain-language INFO/HELP content for the Hardware Builder.
//
// One central map keyed by a stable topic key. Each entry is a short, hobbyist-
// friendly explanation: what the thing is + the practical effect on a mini-sumo
// robot. Keep bodies to 1-3 short sentences. The <Info topic="..."/> component
// (components/Info.tsx) renders these in an on-brand popover; SliderField takes
// an optional `info` prop that points at one of these keys.
//
// Keys fall into three loose groups:
//   - field keys (mass, track_width, …)      one per builder control
//   - step keys (step_body, step_drivetrain) one per interview step intro
//   - legend keys (tof_sensor, line_sensor, com) reused by the 3D legend

export interface HelpTopic {
  title: string;
  body: string;
}

export const HELP: Record<string, HelpTopic> = {
  // --- Chassis / body -------------------------------------------------------
  mass: {
    title: 'Weight',
    body: 'How heavy the whole robot is. Heavier robots are harder to shove out of the ring, but slower to get moving and to turn. Most mini-sumo classes cap this (often 500 g), so aim for the limit without going over.',
  },
  body_length: {
    title: 'Body length',
    body: 'How long the main chassis box is, front to back — not counting the wedge. Longer gives you more room for batteries and motors and resists being spun around; too long and it gets sluggish.',
  },
  width: {
    title: 'Width',
    body: 'How wide the chassis is, side to side. Wider sits more stable and is harder to flip, but a too-wide robot wastes space and is easier to hit from the side.',
  },
  height: {
    title: 'Height',
    body: 'How tall the chassis box is. Lower is almost always better in sumo: it keeps the center of mass down so you are harder to tip and easier to push under the other robot.',
  },
  chassis_friction: {
    title: 'Chassis friction',
    body: 'How grippy the body shell is if it scrapes the mat. Mostly matters when the robot bottoms out or rides up on the opponent — higher means more drag, lower means it slides freely.',
  },
  wheel_friction: {
    title: 'Wheel friction',
    body: "How much grip the tires have on the mat. This is your traction — higher grip means more pushing force and less wheel-spin. It's one of the biggest factors in winning a shoving match.",
  },

  // --- Drivetrain -----------------------------------------------------------
  wheel_radius: {
    title: 'Wheel radius',
    body: 'How big the wheels are. Bigger wheels go faster for the same motor speed but give you less pushing force; smaller wheels are slower but push harder (more torque at the contact point).',
  },
  track_width: {
    title: 'Track width',
    body: 'The distance between your two wheels, left to right. Wider is more stable and harder to tip; narrower turns tighter and spins in place faster.',
  },
  wheel_x_offset: {
    title: 'Wheel front/back offset',
    body: 'How far forward or back the wheels sit under the body. Centering them balances the robot; moving them shifts weight onto the front (better wedge bite) or rear (less likely to nose-dive).',
  },
  max_torque: {
    title: 'Pushing power (max torque)',
    body: 'The strongest twist the motors can apply to the wheels. More torque means more raw pushing force in a shoving match — until the tires lose grip and spin.',
  },
  max_omega: {
    title: 'Wheel spin speed (max omega)',
    body: 'How fast the wheels can spin at full throttle. Combined with wheel size this sets top speed. Faster gets you to the opponent and to good angles first, but is harder to control near the edge.',
  },

  // --- Wedge ----------------------------------------------------------------
  wedge: {
    title: 'Wedge / plow',
    body: 'A sloped blade across the front of the robot. Its job is to slide UNDER the opponent and scoop them up so their wheels lose grip — then you push them out. A good wedge often decides matches.',
  },
  wedge_length: {
    title: 'Wedge length',
    body: 'How far the wedge sticks out in front of the body. Longer reaches under the opponent sooner, but adds weight up front and can dig into the mat.',
  },
  wedge_low_height: {
    title: 'Wedge low edge (front tip)',
    body: 'How high the leading tip of the wedge sits off the mat. Lower gets under the opponent more easily — ideally almost scraping the floor. This is the edge that does the scooping.',
  },
  wedge_high_height: {
    title: 'Wedge high edge (back of wedge)',
    body: 'How tall the wedge is where it meets the body. Together with the low edge this sets the slope of the ramp: a gentler slope lifts opponents more smoothly.',
  },

  // --- Center of mass -------------------------------------------------------
  com: {
    title: 'Center of mass (CoM)',
    body: 'The single balance point of the robot — where all its weight effectively acts (the magenta ⊕ marker). Keeping it LOW makes you hard to flip; keeping it FORWARD presses the wedge down for a better scoop. Get this wrong and the robot tips or pops a wheelie.',
  },
  com_x: {
    title: 'Center of mass · forward/back',
    body: 'Shifts the balance point toward the front or rear. Forward presses the wedge down and grips the front wheels; too far forward and it can nose-dive.',
  },
  com_y: {
    title: 'Center of mass · left/right',
    body: 'Shifts the balance point side to side. Keep this near center (0) so the robot drives straight and pushes evenly with both wheels.',
  },
  com_z: {
    title: 'Center of mass · up/down',
    body: 'How high the balance point sits. Lower is better in sumo — a low center of mass makes the robot much harder to tip over or flip.',
  },

  // --- Distance / ToF sensors ----------------------------------------------
  tof_sensor: {
    title: 'Distance (ToF) sensor',
    body: 'A small "eye" that measures how far away the opponent is by timing a beam of light (Time-of-Flight). This is how the robot spots and locks onto the other robot to charge it.',
  },
  tof_mount: {
    title: 'Sensor mount position',
    body: 'Where the sensor is bolted on the body (forward/back, left/right, up/down). Mounting matters: too low and it sees the mat, too far back and the body blocks its view.',
  },
  tof_angle: {
    title: 'Facing angle (yaw)',
    body: 'Which way the sensor points. 0° looks straight ahead; angling sensors left and right gives the robot a wider field of view so the opponent can\'t sneak around the side.',
  },
  tof_range: {
    title: 'Sensor range',
    body: 'The farthest distance the sensor can still detect the opponent. Longer range spots the enemy sooner from across the ring; beyond the range it reads "nothing there".',
  },

  // --- Line / edge sensors --------------------------------------------------
  line_sensor: {
    title: 'Line (edge) sensor',
    body: 'A downward-looking sensor on the underside that sees the white border ring of the dohyo. It is your "don\'t fall off" alarm — when it spots white, the robot backs away from the edge. Without one, the robot can drive itself out.',
  },
  line_mount: {
    title: 'Line sensor position',
    body: 'Where the edge sensor sits on the underside (forward/back, left/right). Put them near the front corners so the robot gets the most warning before a wheel crosses the line.',
  },

  // --- Dohyo (ring) ---------------------------------------------------------
  dohyo_radius: {
    title: 'Ring radius',
    body: 'How big the dohyo (the round arena) is, measured from the center to the edge. A bigger ring gives more room to maneuver; a standard mini-sumo ring is about 0.77 m across.',
  },
  dohyo_border: {
    title: 'Border width',
    body: 'How thick the painted white ring around the edge is. This is the band your line sensors look for — wider gives the robot a bit more reaction room before the true edge.',
  },

  // --- Validation / contract -----------------------------------------------
  obs_dim: {
    title: 'Observation size (obs dim)',
    body: 'How many numbers the robot\'s brain reads each step (sensor readings, speeds, etc.). Adding or removing sensors changes this. A trained model only works on the exact size it was trained for.',
  },
  action_dim: {
    title: 'Action size (action dim)',
    body: 'How many different moves the robot can choose from each step (drive combinations for the two wheels). Like obs size, a trained model is locked to its action size.',
  },
  signature: {
    title: 'Observation signature',
    body: 'A short fingerprint of this exact sensor/observation layout. Two robots with the same signature can share a trained brain; if it changes, old models no longer fit and you must retrain or fine-tune.',
  },
  finetune: {
    title: 'Fine-tune candidates',
    body: 'Already-trained models whose observation/action sizes match this robot. You can start from one of these and fine-tune instead of training from scratch — a big time saver.',
  },

  // --- Opponent rule-DSL: concepts -----------------------------------------
  opp_rules: {
    title: 'Behavior rules',
    body: 'An ordered list of IF→THEN rules. Each tick the robot checks them top to bottom and runs the FIRST rule whose condition is true. If none match, the DEFAULT action runs.',
  },
  opp_default: {
    title: 'Default action',
    body: 'What the robot does when no rule above it matches this tick. A spinning search (spin_left/right) is a common default so it keeps hunting for the enemy.',
  },
  opp_when: {
    title: 'Condition (WHEN)',
    body: 'The trigger for a rule. Pick a sensor predicate, optionally combine several with ALL (every one true) or ANY (at least one true), and a NOT toggle to invert it.',
  },
  opp_combine: {
    title: 'Combine: ALL / ANY',
    body: 'ALL fires only when every chosen predicate is true at once. ANY fires when at least one is true. With a single predicate this does not matter.',
  },
  opp_not: {
    title: 'NOT (invert)',
    body: 'Flips the whole condition: the rule fires when the condition is FALSE instead of true. E.g. NOT front_hit = "the front sensor does NOT see the enemy".',
  },
  opp_timer: {
    title: 'Timer (every N)',
    body: 'Fires once every N control ticks regardless of sensors — useful for a periodic twitch or to break out of a stalemate. N is a whole number of ticks (about 50 ticks ≈ 1 second).',
  },

  // --- Opponent rule-DSL: predicates ---------------------------------------
  pred_front_hit: {
    title: 'front_hit',
    body: "The front-center sensor beam sees the enemy straight ahead — your cue to charge.",
  },
  pred_left_hit: {
    title: 'left_hit',
    body: 'The front-LEFT sensor beam sees the enemy, i.e. it is off to your front-left.',
  },
  pred_right_hit: {
    title: 'right_hit',
    body: 'The front-RIGHT sensor beam sees the enemy, i.e. it is off to your front-right.',
  },
  pred_side_left_hit: {
    title: 'side_left_hit',
    body: 'The LEFT side sensor sees the enemy beside you on the left.',
  },
  pred_side_right_hit: {
    title: 'side_right_hit',
    body: 'The RIGHT side sensor sees the enemy beside you on the right.',
  },
  pred_edge_left: {
    title: 'edge_left',
    body: 'The left line sensor is over the white border — your left wheel is near the edge, back off that way before you fall out.',
  },
  pred_edge_right: {
    title: 'edge_right',
    body: 'The right line sensor is over the white border — your right wheel is near the edge.',
  },
  pred_no_target: {
    title: 'no_target',
    body: 'None of the sensors see the enemy. You have lost it — usually time to spin and search.',
  },

  // --- Opponent rule-DSL: actions ------------------------------------------
  act_forward: { title: 'forward', body: 'Drive straight ahead at full speed (both wheels forward).' },
  act_reverse: { title: 'reverse', body: 'Back straight up (both wheels reverse) — good to retreat from the edge.' },
  act_spin_left: { title: 'spin_left', body: 'Rotate in place counter-clockwise (left). Used to search or to face the enemy.' },
  act_spin_right: { title: 'spin_right', body: 'Rotate in place clockwise (right). Used to search or to face the enemy.' },
  act_arc_left: { title: 'arc_left', body: 'Curve forward and to the left (left wheel slower) — chase an enemy off to your left.' },
  act_arc_right: { title: 'arc_right', body: 'Curve forward and to the right (right wheel slower) — chase an enemy off to your right.' },
  act_stop: { title: 'stop', body: 'Hold still (both wheels stopped).' },

  opp_hardware: {
    title: 'Opponent hardware',
    body: 'The chassis this opponent fights on. The enemy body, wheels, wedge, mass and motor caps all come from this saved spec at battle time — so pairing a behavior with a heavy or fast preset changes how it actually fights.',
  },

  // --- Opponent: behavior source + presets ---------------------------------
  opp_behavior_source: {
    title: 'Behavior source',
    body: 'Where this opponent\'s brain comes from. BUILT-IN picks one of our zoo controllers (dodger, rammer, novamax, …). CUSTOM RULES lets you author your own IF→THEN behavior. Either one is crossed with the hardware you built in step 1.',
  },
  opp_zoo: {
    title: 'Built-in behavior',
    body: 'One of our hand-written zoo controllers — the same scripted bots the agent trains and is evaluated against. Drop one onto any chassis (e.g. dodger on a heavy body = a "Heavy Dodger").',
  },
  hardware_preset: {
    title: 'Hardware preset',
    body: 'A ready-made chassis you can drop in instead of editing every field. NovaMax is faithful to our reference kit-bot; the others are archetypes (wide pusher, light speedster, heavy rammer, balanced disc). Picking one replaces the current spec; you can still tweak it after.',
  },

  // --- Mini-sumo class limits ----------------------------------------------
  mini_sumo: {
    title: 'Mini-sumo class limits',
    body: 'Regulation mini-sumo robots must weigh at most 500 g and fit inside a 10 x 10 cm box. The chassis MASS, LENGTH and WIDTH below are capped to those limits — an over-limit spec is rejected on save.',
  },

  // --- Zoo behaviors (one per controller) ----------------------------------
  zoo_novamax: {
    title: 'novamax',
    body: 'Our reference bot: tracks the enemy and charges with a steady, well-rounded push. A solid all-round benchmark opponent.',
  },
  zoo_rammer: {
    title: 'rammer',
    body: 'A straight-line charger — locks onto the enemy and drives hard, betting on momentum to push you out.',
  },
  zoo_dodger: {
    title: 'dodger',
    body: 'Evasive: circles and side-steps to avoid head-on hits, trying to make you over-commit and self-out near the edge.',
  },
  zoo_spinner: {
    title: 'spinner',
    body: 'Spins in place to search, then darts at the enemy when a sensor catches it — unpredictable angles of attack.',
  },
  zoo_wedger: {
    title: 'wedger',
    body: 'A wedge-pusher: lines up to get its plow under you, then shovels you toward the edge.',
  },
  zoo_charger: {
    title: 'charger',
    body: 'Aggressive rushing behavior — commits early and hard toward the last seen enemy position.',
  },
  zoo_tracker: {
    title: 'tracker',
    body: 'Patient pursuer: keeps the enemy centred in its sensors and follows, looking for a clean pushing angle.',
  },
  zoo_feinter: {
    title: 'feinter',
    body: 'Feints and bait moves to draw you into a mistake. Held out of the standard training mix — an eval-only test of generalization.',
  },
  zoo_orbiter: {
    title: 'orbiter',
    body: 'Orbits around the enemy looking for a flank. Held out of the standard training mix — an eval-only test of generalization.',
  },

  // --- Interview step intros (one sentence each) ---------------------------
  step_body: {
    title: 'Robot body',
    body: 'Set the size and weight of the main chassis box. This is the core of the robot — everything else bolts onto it.',
  },
  step_drivetrain: {
    title: 'Drivetrain',
    body: 'Set up the wheels and motors that move and turn the robot. This decides your speed and pushing power.',
  },
  step_wedge: {
    title: 'Wedge / plow',
    body: 'Decide whether the robot has a sloped front blade to scoop opponents up, and shape it.',
  },
  step_com: {
    title: 'Center of mass',
    body: 'Place the robot\'s balance point. Low and forward keeps it from flipping and presses the wedge down.',
  },
  step_tof: {
    title: 'Forward sensors',
    body: 'Add the distance sensors the robot uses to find and track the opponent.',
  },
  step_line: {
    title: 'Edge sensors',
    body: 'Add the underside sensors that spot the white border ring so the robot doesn\'t drive itself out.',
  },
  step_dohyo: {
    title: 'Dohyo ring',
    body: 'Set the size of the arena the robot fights in.',
  },
};

export type HelpKey = keyof typeof HELP;
