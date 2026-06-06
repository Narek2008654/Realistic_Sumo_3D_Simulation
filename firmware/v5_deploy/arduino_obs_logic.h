// arduino_obs_logic.h — byte-for-byte mirror of sumo_env.py's
// _update_last_seen / _sensor_strength.
//
// IMPORTANT: any change to the Python state machine must be matched
// here on the same commit, or sim-to-real will silently drift.
//
// Constants (LAST_SEEN_HYSTERESIS_RATIO, LAST_SEEN_DECAY_STEPS,
// ENEMY_FAR_DIST) MUST equal the Python values. Steps are integers;
// LAST_SEEN_DECAY_STEPS = round(LAST_SEEN_DECAY_SECONDS / STEP_DT_SECONDS)
// in Python, hard-coded here for AVR.

#pragma once

#include <math.h>
#include <stdint.h>

namespace mini_sumo_obs {

// Match sumo_env.py.
constexpr float    ENEMY_FAR_DIST                 = 0.80f;
constexpr float    LAST_SEEN_HYSTERESIS_RATIO     = 1.1f;
constexpr uint16_t LAST_SEEN_DECAY_STEPS          = 12;   // 0.5 s @ 24 Hz

// Persistent state (one instance per match — call reset() once at start).
struct LastSeenState {
    float    dir;             // -1.0 / 0.0 / +1.0  (left / front / right)
    uint16_t steps_since_hit; // saturates at LAST_SEEN_DECAY_STEPS
};

inline void reset(LastSeenState &s) {
    s.dir = 0.0f;
    s.steps_since_hit = 0;
}

// Convert raw distance (m) to strength in [0, 1]. Closer = stronger.
// NaN, +inf, or any value >= ENEMY_FAR_DIST count as "no hit" → 0.
static inline float strength(float d) {
    if (isnan(d) || d >= ENEMY_FAR_DIST) return 0.0f;
    if (d <= 0.0f)                       return 1.0f;
    return 1.0f - (d / ENEMY_FAR_DIST);
}

// Update last-seen direction in place. distances in metres, order
// {front, left, right} — same as the Python observation.
inline void update_last_seen(LastSeenState &s,
                             float front_m,
                             float left_m,
                             float right_m) {
    const float s0 = strength(front_m);
    const float s1 = strength(left_m);
    const float s2 = strength(right_m);
    float max_s = s0;
    if (s1 > max_s) max_s = s1;
    if (s2 > max_s) max_s = s2;

    // All sensors at max range → start the decay timer.
    if (max_s <= 0.0f) {
        if (s.steps_since_hit < LAST_SEEN_DECAY_STEPS) {
            s.steps_since_hit++;
        }
        if (s.steps_since_hit >= LAST_SEEN_DECAY_STEPS) {
            s.dir = 0.0f;
        }
        return;
    }
    s.steps_since_hit = 0;

    // Map current latched direction to a sensor index. 0=front, 1=left, 2=right.
    uint8_t prev_idx;
    if      (s.dir < -0.5f) prev_idx = 1;
    else if (s.dir >  0.5f) prev_idx = 2;
    else                    prev_idx = 0;
    const float prev_s = (prev_idx == 0) ? s0
                       : (prev_idx == 1) ? s1
                       :                    s2;

    // Tie-break (ordered to match Python's `tied[0]` over [front, left, right]):
    //   1. if the previous winner is tied with the max, keep it (sticky)
    //   2. else first tied index in declared order
    uint8_t winner;
    if      (prev_idx == 0 && s0 == max_s) winner = 0;
    else if (prev_idx == 1 && s1 == max_s) winner = 1;
    else if (prev_idx == 2 && s2 == max_s) winner = 2;
    else if (s0 == max_s)                  winner = 0;
    else if (s1 == max_s)                  winner = 1;
    else                                   winner = 2;

    // Hysteresis: only switch if max_s > 1.1 × prev_s.
    if (winner != prev_idx) {
        if (max_s <= LAST_SEEN_HYSTERESIS_RATIO * prev_s) {
            return;
        }
    }
    s.dir = (winner == 0) ? 0.0f
          : (winner == 1) ? -1.0f
          :                  1.0f;
}

// ====================================================================
// Run 2: prev_action latch (obs[6:8]).
//
// Persist the last predict_motors output between loop iterations and
// feed it back as input. The Python env latches raw_left/raw_right
// (post-clip, pre-deadzone) so the firmware-side equivalent is the
// raw output of the previous inference call — no deadzone or latency
// modelling required on hardware.
// ====================================================================

struct PrevActionState {
    float left;
    float right;
};

inline void reset(PrevActionState &s) {
    s.left = 0.0f;
    s.right = 0.0f;
}

inline void update(PrevActionState &s, float new_left, float new_right) {
    s.left = new_left;
    s.right = new_right;
}

// ====================================================================
// Run 2: engagement_timer (obs[8]).
//
// Counts consecutive ticks the front laser sees the enemy below
// ENGAGEMENT_FRONT_THRESHOLD on the normalised distance scale. The
// normalised value (count / ENGAGEMENT_MAX_STEPS, clamped to [0, 1])
// goes to the policy. Constants MUST equal the Python values in
// sumo_env.py or sim2real diverges.
// ====================================================================

constexpr float    ENGAGEMENT_FRONT_THRESHOLD = 0.15f;
constexpr uint16_t ENGAGEMENT_MAX_STEPS       = 30;

struct EngagementTimerState {
    uint16_t count;
};

inline void reset(EngagementTimerState &s) {
    s.count = 0;
}

inline void update(EngagementTimerState &s, float front_norm) {
    if (front_norm < ENGAGEMENT_FRONT_THRESHOLD) {
        // Saturate at MAX_STEPS to avoid uint16 overflow on a stuck
        // engagement (e.g. the firmware sits in a corner with the
        // enemy locked indefinitely). The Python counter is unbounded
        // but normalises identically, so the float seen by the policy
        // matches sim exactly.
        if (s.count < ENGAGEMENT_MAX_STEPS) s.count++;
    } else {
        s.count = 0;
    }
}

inline float normalized(const EngagementTimerState &s) {
    const float v = (float)s.count / (float)ENGAGEMENT_MAX_STEPS;
    return v > 1.0f ? 1.0f : v;
}

// ====================================================================
// Run 6: yaw-rate proxy (obs[9]). Mirrors sumo_env.py's accumulator
// EXACTLY: value += (left - right) * 0.1, then *= 0.9, then clamp to
// [-1, +1]. Two floats and a multiply per tick — no IMU required.
// Saturates at ±1.0 during sustained tank-spin (steady-state of
// (v + 1.2*0.1) * 0.9 has fixed point at 1.08, which the clamp pins).
// ====================================================================
struct YawRateProxyState {
    float value;
};

inline void reset(YawRateProxyState &s) { s.value = 0.0f; }

inline void update(YawRateProxyState &s, float left_cmd, float right_cmd) {
    s.value = (s.value + (left_cmd - right_cmd) * 0.1f) * 0.9f;
    if (s.value > 1.0f)  s.value = 1.0f;
    if (s.value < -1.0f) s.value = -1.0f;
}

// ====================================================================
// Run 9: IR delta features (obs[10], obs[11]).
//
// Two floats of persistent state: the previous frame's normalised
// IR readings. The new obs features are temporal derivatives, scaled
// and clipped to [-1, +1] identically to sumo_env.py's _build_obs.
//
//   front_ir_delta   = clip((front_norm - prev_front)   * 2, ±1)
//   lateral_ir_delta = clip((min_lat   - prev_min_lat)  * 2, ±1)
//
// At reset, prev values = 1.0 (the "no hit" encoding), so the first
// frame produces a zero delta. Each tick: compute delta, then cache
// the new value for next call. ~4 multiplies and 2 compares total —
// no measurable cost at 25 Hz.
// ====================================================================

struct IrDeltaState {
    float prev_front_norm;
    float prev_min_lateral;
};

inline void reset(IrDeltaState &s) {
    s.prev_front_norm = 1.0f;
    s.prev_min_lateral = 1.0f;
}

// Returns (front_delta, lateral_delta) via out-params; updates state.
inline void update(
    IrDeltaState &s,
    float front_norm, float left_norm, float right_norm,
    float &out_front_delta, float &out_lateral_delta
) {
    const float min_lat = (left_norm < right_norm) ? left_norm : right_norm;
    float fd = (front_norm - s.prev_front_norm) * 2.0f;
    float ld = (min_lat   - s.prev_min_lateral) * 2.0f;
    if (fd >  1.0f) fd =  1.0f;
    if (fd < -1.0f) fd = -1.0f;
    if (ld >  1.0f) ld =  1.0f;
    if (ld < -1.0f) ld = -1.0f;
    out_front_delta   = fd;
    out_lateral_delta = ld;
    s.prev_front_norm  = front_norm;
    s.prev_min_lateral = min_lat;
}

}  // namespace mini_sumo_obs
