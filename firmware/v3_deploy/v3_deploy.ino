// v3_deploy.ino — Mini Sumo Arduino Nano, runs dqn_3d_bc_actor_best.pt
// (the v3 model trained on the realistic 3D PyBullet sim with all
// physics fixes: anisotropic wheel friction, free-spin idle, -50
// self-out penalty, anti-flicker reward, BC pretrain on CombatPolicy).
//
// Pipeline (~24 Hz target):
//   read 3x VL53L0X -> build 12D obs -> predict_action -> drive motors
//
// Hardware (matches user's wiring):
//   VL53L0X x3 on I2C, XSHUT pins to set addresses on boot:
//     front  XSHUT=D2  addr=0x30
//     left   XSHUT=D3  addr=0x31
//     right  XSHUT=D4  addr=0x32
//   TB6612 motor driver:
//     left motor  (A): IN1=D10, IN2=D9,  PWM=D11
//     right motor (B): IN1=D6,  IN2=D7,  PWM=D5
//     STBY = D8 (HIGH = enabled)
//   QTR-1A line sensors (analog, active-LOW over white border):
//     rear-left  = A0
//     rear-right = A1
//
// Action space is {-1, 0, +1} per wheel, so no PWM modulation needed —
// PWM pins held HIGH for full speed, direction pins switch sign.

#include <Wire.h>
#include <VL53L0X.h>

#include "neural_net_v6_3d.h"   // mini_sumo_ai::predict_action, ACTION_MAP
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

// Convert mm reading to obs distance in metres. OOR -> 0.8 m so the
// policy sees max range (matches sim's ENEMY_FAR_DIST convention).
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
// QTR-1A outputs LOW over white (reflective) and HIGH over black mat.
// Sumo border ring is white -> reading below LINE_THRESHOLD = on line.
// Tune LINE_THRESHOLD with the print in setup() if needed.
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
    last_valid_sensor_ms = millis();

    Serial.println(F("ready (v3, 3D-trained)"));
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

    const uint32_t now = millis();
    if (any_forward_valid(front_m, left_m, right_m)) {
        last_valid_sensor_ms = now;
    }

    mini_sumo_obs::update_last_seen(last_seen_state, front_m, left_m, right_m);

    float out_l = 0.0f, out_r = 0.0f;
    const bool watchdog_fired = (now - last_valid_sensor_ms) > WATCHDOG_TIMEOUT_MS;

    if (watchdog_fired) {
        out_l = SEARCH_LEFT;
        out_r = SEARCH_RIGHT;
    } else {
        const float fn = norm_distance(front_m);
        const float ln = norm_distance(left_m);
        const float rn = norm_distance(right_m);

        mini_sumo_obs::update(engagement_timer, fn);
        float front_delta = 0.0f, lateral_delta = 0.0f;
        mini_sumo_obs::update(ir_delta_state, fn, ln, rn, front_delta, lateral_delta);

        const float obs[mini_sumo_ai::INPUT_DIM] = {
            fn, ln, rn,
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

        mini_sumo_obs::update(prev_action,    out_l, out_r);
        mini_sumo_obs::update(yaw_rate_proxy, out_l, out_r);
    }

    drive_left(out_l);
    drive_right(out_r);

    // Loop pacing: aim ~25 Hz to match sumo_env.STEP_DT_SECONDS = 1/24.
    // VL53L0X budget = 33 ms per sensor reading dominates; the delay
    // here is small headroom only.
    delay(5);
}
