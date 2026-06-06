// v5_deploy.ino — Mini Sumo Arduino Nano, runs ppo_ent0_best.pt
// (the 21-D discrete-PPO champion: BC warm start + curriculum + zero-entropy
// sharpening, trained on the realistic 3D PyBullet sim, branch pol/ppo).
// Beats Stage-A head-to-head 65% with ~3x fewer self-outs and far better
// generalization (83% held-out); meets the 65%-overall / 55%-novamax bar.
// The PPO actor head argmax = the action, so it uses the same discrete
// export path as the DQN. NOTE: trained WITH the hardcoded safety override,
// which is replicated in loop() below.
//
// DIFFERENCE FROM v3: the policy now consumes a 21-D observation = the 3
// VL53L0X distance channels STACKED over the last K=4 frames (oldest
// first) followed by the 9 single-frame engineered features. A small
// ring buffer of the normalized distances reproduces sumo_env's
// RawDistanceStack wrapper byte-for-byte (see obs_stack.py). The old
// 12-D model is untouched in ../v3_deploy/.
//
// Pipeline (~24 Hz target):
//   read 3x VL53L0X -> push distance ring -> build 21D obs -> predict -> drive
//
// Hardware / wiring identical to v3_deploy (see that file's header).
//   VL53L0X x3 on I2C (XSHUT D2/D3/D4 -> addr 0x30/0x31/0x32)
//   TB6612: left IN1=D10 IN2=D9 PWM=D11 ; right IN1=D6 IN2=D7 PWM=D5 ; STBY=D8
//   QTR-1A rear line sensors: rear-left=A0 rear-right=A1

#include <Wire.h>
#include <VL53L0X.h>

#include "neural_net_v6_3d.h"   // mini_sumo_ai::predict_action (INPUT_DIM=21)
#include "arduino_obs_logic.h"  // mini_sumo_obs:: state machines

// =====================================================================
// Pins
// =====================================================================
#define XSHUT_FRONT 2
#define XSHUT_LEFT  3
#define XSHUT_RIGHT 4

const int motorA1 = 10;  // left  IN1
const int motorA2 = 9;   // left  IN2
const int motorB1 = 6;   // right IN1
const int motorB2 = 7;   // right IN2
#define pwmA 11
#define pwmB 5
#define stby 8

VL53L0X frontSensor, leftSensor, rightSensor;

// =====================================================================
// Sensor read with outlier rejection (keeps last good reading on glitch)
// =====================================================================
#define OUTLIER_MAX 8000  // VL53L0X returns 8190 for out-of-range

uint16_t frontLastValid = OUTLIER_MAX;
uint16_t leftLastValid  = OUTLIER_MAX;
uint16_t rightLastValid = OUTLIER_MAX;

static uint16_t filteredRead(VL53L0X& sensor, uint16_t& lastValid) {
    uint16_t raw = sensor.readRangeContinuousMillimeters();
    if (raw == 0) {
        return lastValid;            // I2C read failed, hold previous
    }
    if (raw > OUTLIER_MAX) {
        lastValid = OUTLIER_MAX;     // out-of-range = no target
        return OUTLIER_MAX;
    }
    lastValid = raw;
    return raw;
}

static inline float mm_to_obs_m(uint16_t mm) {
    if (mm >= (uint16_t)(mini_sumo_obs::ENEMY_FAR_DIST * 1000.0f)) {
        return mini_sumo_obs::ENEMY_FAR_DIST;
    }
    return (float)mm * 0.001f;
}

static inline float norm_distance(float d_m) {
    if (d_m >= mini_sumo_obs::ENEMY_FAR_DIST) return 1.0f;
    if (d_m <= 0.0f) return 0.0f;
    return d_m / mini_sumo_obs::ENEMY_FAR_DIST;
}

// =====================================================================
// Line sensors — QTR-1A analog phototransistors at the rear.
// =====================================================================
#define LINE_L_PIN  A0
#define LINE_R_PIN  A1
constexpr int LINE_THRESHOLD = 500;  // ADC count, ~mid of 0..1023

static inline bool read_line_l() { return analogRead(LINE_L_PIN) < LINE_THRESHOLD; }
static inline bool read_line_r() { return analogRead(LINE_R_PIN) < LINE_THRESHOLD; }

// =====================================================================
// Motor drive — action_map entries are in {-1, 0, +1}
// =====================================================================
static void drive_left(float cmd) {
    if (cmd > 0.5f)        { digitalWrite(motorA1, HIGH); digitalWrite(motorA2, LOW);  }
    else if (cmd < -0.5f)  { digitalWrite(motorA1, LOW);  digitalWrite(motorA2, HIGH); }
    else                   { digitalWrite(motorA1, LOW);  digitalWrite(motorA2, LOW);  }
}

static void drive_right(float cmd) {
    if (cmd > 0.5f)        { digitalWrite(motorB1, HIGH); digitalWrite(motorB2, LOW);  }
    else if (cmd < -0.5f)  { digitalWrite(motorB1, LOW);  digitalWrite(motorB2, HIGH); }
    else                   { digitalWrite(motorB1, LOW);  digitalWrite(motorB2, LOW);  }
}

// =====================================================================
// Watchdog: if nothing in forward arc for a while, do a search-spin
// =====================================================================
constexpr float    VALID_RANGE_M       = 0.80f * mini_sumo_obs::ENEMY_FAR_DIST;
constexpr uint16_t WATCHDOG_TIMEOUT_MS = 1500;
constexpr float    SEARCH_LEFT  = +1.0f;
constexpr float    SEARCH_RIGHT = -1.0f;

uint32_t last_valid_sensor_ms = 0;

static inline bool any_forward_valid(float f, float l, float r) {
    return (f < VALID_RANGE_M) || (l < VALID_RANGE_M) || (r < VALID_RANGE_M);
}

// =====================================================================
// Distance ring buffer = sumo_env's RawDistanceStack (obs_stack.py).
// Holds the last K=4 frames of NORMALIZED (front,left,right), oldest at
// index 0, newest at index K-1. Seeded by replicating frame 0 on the
// first loop (matches the wrapper's reset semantics). The stacked obs is
// [ring[0..3] flattened, then the 9 engineered features].
// =====================================================================
constexpr uint8_t STACK_K = 4;     // must equal obs_stack.DEFAULT_STACK_K
float dist_ring[STACK_K][3];
bool  ring_init = false;

static void push_dist_ring(float fn, float ln, float rn) {
    if (!ring_init) {
        for (uint8_t k = 0; k < STACK_K; ++k) {
            dist_ring[k][0] = fn; dist_ring[k][1] = ln; dist_ring[k][2] = rn;
        }
        ring_init = true;
        return;
    }
    for (uint8_t k = 0; k < STACK_K - 1; ++k) {   // shift oldest out
        dist_ring[k][0] = dist_ring[k + 1][0];
        dist_ring[k][1] = dist_ring[k + 1][1];
        dist_ring[k][2] = dist_ring[k + 1][2];
    }
    dist_ring[STACK_K - 1][0] = fn;               // newest in
    dist_ring[STACK_K - 1][1] = ln;
    dist_ring[STACK_K - 1][2] = rn;
}

// =====================================================================
// Persistent obs state
// =====================================================================
mini_sumo_obs::LastSeenState        last_seen_state;
mini_sumo_obs::PrevActionState      prev_action;
mini_sumo_obs::EngagementTimerState engagement_timer;
mini_sumo_obs::YawRateProxyState    yaw_rate_proxy;
mini_sumo_obs::IrDeltaState         ir_delta_state;

// =====================================================================
// Setup
// =====================================================================
void setup() {
    Serial.begin(115200);

    pinMode(motorA1, OUTPUT); pinMode(motorA2, OUTPUT);
    pinMode(motorB1, OUTPUT); pinMode(motorB2, OUTPUT);
    pinMode(pwmA,    OUTPUT); pinMode(pwmB,    OUTPUT);
    pinMode(stby,    OUTPUT);

    digitalWrite(pwmA, HIGH);   // full speed (action space is {-1,0,+1})
    digitalWrite(pwmB, HIGH);
    digitalWrite(stby, HIGH);
    drive_left(0.0f);
    drive_right(0.0f);

    Wire.begin();

    pinMode(XSHUT_FRONT, OUTPUT);
    pinMode(XSHUT_LEFT,  OUTPUT);
    pinMode(XSHUT_RIGHT, OUTPUT);
    digitalWrite(XSHUT_FRONT, LOW);
    digitalWrite(XSHUT_LEFT,  LOW);
    digitalWrite(XSHUT_RIGHT, LOW);
    delay(10);

    digitalWrite(XSHUT_FRONT, HIGH); delay(10);
    frontSensor.init(); frontSensor.setAddress(0x30);
    frontSensor.setSignalRateLimit(0.5);
    frontSensor.setMeasurementTimingBudget(33000);

    digitalWrite(XSHUT_LEFT, HIGH); delay(10);
    leftSensor.init();  leftSensor.setAddress(0x31);
    leftSensor.setSignalRateLimit(0.5);
    leftSensor.setMeasurementTimingBudget(33000);

    digitalWrite(XSHUT_RIGHT, HIGH); delay(10);
    rightSensor.init(); rightSensor.setAddress(0x32);
    rightSensor.setSignalRateLimit(0.5);
    rightSensor.setMeasurementTimingBudget(33000);

    frontSensor.startContinuous();
    leftSensor.startContinuous();
    rightSensor.startContinuous();

    // Verify model integrity — refuse to drive if weights corrupted
    const uint8_t mismatches = mini_sumo_ai::verify_self_test();
    Serial.print(F("model self-test mismatches: "));
    Serial.println(mismatches);
    if (mismatches > 0) {
        Serial.println(F("FATAL: re-flash neural_net_v6_3d.h"));
        while (1) { delay(1000); }
    }

    mini_sumo_obs::reset(last_seen_state);
    mini_sumo_obs::reset(prev_action);
    mini_sumo_obs::reset(engagement_timer);
    mini_sumo_obs::reset(yaw_rate_proxy);
    mini_sumo_obs::reset(ir_delta_state);
    ring_init = false;
    last_valid_sensor_ms = millis();

    Serial.println(F("ready (v5, 21-D PPO champion + safety override)"));
}

// =====================================================================
// Main loop
// =====================================================================
void loop() {
    const uint16_t mmF = filteredRead(frontSensor, frontLastValid);
    const uint16_t mmL = filteredRead(leftSensor,  leftLastValid);
    const uint16_t mmR = filteredRead(rightSensor, rightLastValid);

    const float front_m = mm_to_obs_m(mmF);
    const float left_m  = mm_to_obs_m(mmL);
    const float right_m = mm_to_obs_m(mmR);

    const float fn = norm_distance(front_m);
    const float ln = norm_distance(left_m);
    const float rn = norm_distance(right_m);

    // Update all per-frame state EVERY tick (even during the watchdog) so
    // the ring + state machines never go stale and match the sim 1:1.
    push_dist_ring(fn, ln, rn);
    mini_sumo_obs::update_last_seen(last_seen_state, front_m, left_m, right_m);
    mini_sumo_obs::update(engagement_timer, fn);
    float front_delta = 0.0f, lateral_delta = 0.0f;
    mini_sumo_obs::update(ir_delta_state, fn, ln, rn, front_delta, lateral_delta);

    const uint32_t now = millis();
    if (any_forward_valid(front_m, left_m, right_m)) {
        last_valid_sensor_ms = now;
    }
    const bool watchdog_fired = (now - last_valid_sensor_ms) > WATCHDOG_TIMEOUT_MS;

    float out_l = 0.0f, out_r = 0.0f;
    if (watchdog_fired) {
        out_l = SEARCH_LEFT;
        out_r = SEARCH_RIGHT;
    } else {
        // 21-D stacked observation: 4x(front,left,right) oldest-first,
        // then the 9 single-frame engineered features.
        const float obs[mini_sumo_ai::INPUT_DIM] = {
            dist_ring[0][0], dist_ring[0][1], dist_ring[0][2],
            dist_ring[1][0], dist_ring[1][1], dist_ring[1][2],
            dist_ring[2][0], dist_ring[2][1], dist_ring[2][2],
            dist_ring[3][0], dist_ring[3][1], dist_ring[3][2],
            last_seen_state.dir,
            read_line_l() ? 1.0f : 0.0f,
            read_line_r() ? 1.0f : 0.0f,
            prev_action.left, prev_action.right,
            mini_sumo_obs::normalized(engagement_timer),
            yaw_rate_proxy.value,
            front_delta, lateral_delta,
        };
        const uint8_t a_idx = mini_sumo_ai::predict_action(obs);
        mini_sumo_ai::action_to_motors(a_idx, out_l, out_r);

        // Hardcoded safety override — the PPO policy was TRAINED with this
        // on (sumo_env._apply_safety_override), so it must run here too or
        // deployed behaviour diverges from training. (1) rear line sensor
        // over the border -> drive inward; (2) net-forward with nothing
        // detected and the target lost -> in-place scan-spin. Observable
        // signals only. SAFETY_CLEAR_NORM = 0.90.
        if (read_line_l() || read_line_r()) {
            out_l = 1.0f; out_r = 1.0f;
        } else if (last_seen_state.dir == 0.0f &&
                   fn > 0.90f && ln > 0.90f && rn > 0.90f &&
                   out_l > 0.5f && out_r > 0.5f) {
            out_l = 1.0f; out_r = -1.0f;
        }
    }

    // prev_action / yaw proxy track the action actually executed (both
    // the policy and the watchdog search-spin), so the next obs is correct.
    mini_sumo_obs::update(prev_action,    out_l, out_r);
    mini_sumo_obs::update(yaw_rate_proxy, out_l, out_r);

    drive_left(out_l);
    drive_right(out_r);

    // Loop pacing: VL53L0X 33 ms budget dominates; small headroom delay.
    delay(5);
}
