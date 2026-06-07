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
