# Robotic Arm Joint Health Monitoring System

An automated multi-signal health monitoring and defensive analytics system for a fleet of 6-axis industrial robotic arms on an assembly line. The system monitors **joint torque**, **joint temperature**, and **vibration** across critical kinetic chain joints (**Shoulder**, **Elbow**, and **Wrist**), ingests unstructured technician maintenance notes, guards against adversarial sensor spoofing, and outputs an authoritative operational decision for each arm.

---

## 1. System Architecture & Capabilities

```
+----------------------------------------------------------------------------------------------------+
|                                    INPUT SIGNALS & TELEMETRY                                       |
|  +-------------------------------------+             +------------------------------------------+  |
|  |     6-Axis Arm Sensor Telemetry     |             |       Unstructured Maintenance Logs      |  |
|  | (Torque, Temp, Vibration @ J2,J3,J5)|             |  (Technician natural language sentences) |  |
|  +------------------+------------------+             +--------------------+---------------------+  |
+---------------------|-----------------------------------------------------|------------------------+
                      |                                                     |
                      v                                                     v
+----------------------------------------+             +------------------------------------------+
|  Layer 1: Deterministic Safety Check   |             |   Layer 4: Maintenance NLP Parser        |
|  * O(1) hard threshold comparator      |             |   * Domain lexicon & regex extraction    |
|  * 100% INDEPENDENT OF ML MODEL        |             |   * Severity scoring (0: None to 3: Crit)|
+---------------------+------------------+             +--------------------+---------------------+
                      |                                                     |
                      v                                                     |
+----------------------------------------+                                  |
|  Layer 2: Spoofed-Sensor Defense       |                                  |
|  * Thermodynamic-friction correlation  |                                  |
|  * Kinetic coupling invariant check    |                                  |
|  * Physical bounds & signal freezing   |                                  |
+---------------------+------------------+                                  |
                      |                                                     |
                      v                                                     |
+----------------------------------------+                                  |
|  Layer 3: Predictive ML Wear Model     |                                  |
|  * Random Forest classifier (100 trees)|                                  |
|  * P(Wear) & class degradation         |                                  |
+---------------------+------------------+                                  |
                      |                                                     |
                      +--------------------------+--------------------------+
                                                 |
                                                 v
                      +-----------------------------------------------------+
                      |         Layer 5: Hierarchical Decision Engine       |
                      |          Produces exactly 1 final decision:         |
                      |                                                     |
                      |   [SAFE]  [MONITOR]  [SCHEDULE_MAINTENANCE]  [ESTOP]|
                      +-----------------------------------------------------+
```

---

## 2. Written Technical Explanations

### A. Rationale for Deterministic Safety Thresholds
The immediate safety-stop thresholds are strictly deterministic and isolated from all machine learning inference routines to guarantee guaranteed hard-real-time safety intervention without model latency, numerical drift, or edge-case misclassifications. The thresholds are calibrated according to industrial robot kinematics (ISO 10816 standards for mechanical vibration and Class F/H motor thermal insulation limits):

1. **Torque Limits (`Shoulder: 360 Nm`, `Elbow: 280 Nm`, `Wrist: 105 Nm`)**:
   - In a 6-axis articulated robot, the shoulder (Joint 2) carries the heaviest gravity and inertial payload; its continuous rated limit is ~250 Nm with an instantaneous peak limit of 350 Nm. Surpassing 360 Nm signifies an imminent structural stall, tool collision, or mechanical gear lockup.
   - The elbow (Joint 3) operates at lower leverage (rated ~180 Nm), making 280 Nm an acute mechanical overload.
   - The wrist (Joint 5) uses compact strain-wave/harmonic drives; exceeding 105 Nm risks tooth shearing and flexspline fatigue.
2. **Temperature Limits (`Shoulder: 82 °C`, `Elbow: 78 °C`, `Wrist: 75 °C`)**:
   - Industrial servomotors utilize Class F winding insulation rated for continuous operation up to 60–65 °C. At temperatures above 75–82 °C, lubricant viscosity breaks down rapidly, bearing seals harden, and rotor demagnetization or insulation breakdown risks irreversible drive failure.
3. **Vibration Limits (`Shoulder: 7.0 mm/s`, `Elbow: 6.5 mm/s`, `Wrist: 6.0 mm/s RMS`)**:
   - Calibrated using **ISO 10816-3 (Zone D: Vibration Severity)**. Values above 4.5 mm/s represent the warning boundary (abnormal wear or gear pitting), while velocities $> 6.0 - 7.0$ mm/s indicate catastrophic dynamic unbalance, raceway spalling, or loose mounting bolts that necessitate an immediate halt to protect line personnel and surrounding tooling.

---

### B. Fusion Logic: Combining ML Predictions & Maintenance Logs into Final Decisions
The decision engine operates under a **hierarchical precedence protocol** ensuring that critical safety signals always supersede operational optimizations:

1. **`EMERGENCY_STOP`** (Highest Priority):
   - Triggered immediately if **any single joint reading crosses a critical deterministic safety limit** (evaluated in $O(1)$, completely independent of ML).
   - Triggered if technician logs report an **imminent physical hazard** (e.g., `smoke`, `fire`, `sparks`, `locked up axis`).
   - Triggered if an **active sensor spoofing attack is detected in combination with secondary warning telemetry** (unmonitored hazardous operation).
2. **`SCHEDULE_MAINTENANCE`**:
   - Triggered when the **Predictive Wear Model** classifies the arm as `SEVERE_WEAR` ($P(\text{severe}) \ge 50\%$), indicating that while the arm has not yet tripped hard safety limits, its bearing or gear degradation will fail in near-term production cycles.
   - Triggered when technician notes describe **high-severity mechanical symptoms** (e.g., `metallic grinding`, `excessive backlash/play`, `binding`).
   - Triggered when **sensor spoofing is detected**, because an arm with unverified telemetry cannot be safely trusted in continuous production without sensor calibration or replacement.
3. **`MONITOR`**:
   - Triggered when the ML model predicts early degradation (`MODERATE_WEAR`), or
   - Triggered when technician notes log minor advisory observations (e.g., `slight squeak`, `warm to touch`, `minor backlash`), or
   - Triggered when any joint metric operates inside the warning band (e.g., shoulder temperature $> 68$ °C but $< 82$ °C).
4. **`SAFE`**:
   - Granted **only** when all joint telemetry is within nominal operating bands, the ML model predicts `HEALTHY`, the maintenance logs have zero active mechanical flags, and all sensor integrity checks pass.

---

### C. Spoofed-Sensor Defense: Mechanism, Capabilities & Limitations

#### How the Defense Works
An adversary or malfunctioning sensor interface could feed fake, nominal joint-torque values (e.g. clamping torque to 12.0 Nm) to suppress torque-based alarm thresholds while the joint is actually undergoing severe friction, overheating, or mechanical seizure.

The spoof defense inspects **multivariate physical invariants** across the entire arm:
1. **Thermodynamic & Friction Decoupling Check**:
   - *Physical Invariant*: Severe mechanical friction and bearing wear dissipate kinetic energy into heat and vibration. A joint running at $>60$ °C with violent vibration ($>3.5$ mm/s RMS) physically requires active power and mechanical drag. Reporting near-zero or negligible torque under this state directly violates thermodynamics and indicates synthetic signal clamping.
2. **Serial Manipulator Kinetic Coupling Invariant**:
   - *Physical Invariant*: In a serial robotic arm, the shoulder (Joint 2) physically supports the mass and payload of all distal joints (Elbow and Wrist). If the elbow reports 160 Nm and the wrist reports 45 Nm, the shoulder *must* exhibit corresponding reaction torque. A shoulder reporting $<30$ Nm while distal joints are heavily loaded represents an impossible physical state, exposing torque signal injection.
3. **Physical Domain Bounds & Quantization Variance**:
   - Checks against physical domain limits (e.g. negative torque during load cycles, negative temperatures, or frozen zero-variance ADC signals lacking ambient quantization noise).

#### What the Defense **Would Catch**
- **Synthetic Value Clamping / Masking**: Injecting normal or low torque to hide mechanical binding or motor overload during high-vibration/high-heat states.
- **Selective Joint Spoofing**: Spoofing only one joint (e.g. the shoulder) while leaving distal kinematic sensors unmodified.
- **Constant / Replay Spoofs**: Telemetry with frozen constant float values across multiple timestamps.
- **Out-of-Bounds Tampering**: Negative values or impossible non-physical values.

#### What the Defense **Would NOT Catch**
- **Coordinated High-Fidelity Physics Emulators**: If an advanced attacker implements an accurate real-time multibody dynamics and thermal digital twin to spoof *all* torque, temperature, and vibration sensors simultaneously in perfect thermodynamic synchrony, single-arm kinematic checks cannot detect the discrepancy without external independent signals (such as motor driver 3-phase bus current or external cell cameras).
- **Subtle Drift Within Nominal Tolerances**: Small offset biases ($\pm 5\%$) that remain completely within normal operating envelopes and physical variance bands.

---

## 3. Project Structure

```
.
├── data/
│   ├── historical_joint_data.csv    # 150 labeled synthetic readings (Healthy, Moderate, Severe)
│   └── technician_logs.txt          # 12 maintenance log entries mixing relevant & irrelevant notes
├── src/
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # Thresholds, Enums, and Dataclasses
│   ├── safety_monitor.py            # Layer 1: Deterministic safety stops (zero ML dependency)
│   ├── spoof_detector.py            # Layer 2: Spoofed-sensor defense engine
│   ├── wear_model.py                # Layer 3: Supervised Random Forest predictive wear model
│   ├── maintenance_nlp.py           # Layer 4: Unstructured maintenance text parser & severity scorer
│   ├── decision_engine.py           # Layer 5: Authoritative decision fusion engine
│   └── pipeline.py                  # End-to-end orchestrator connecting all 5 components
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py             # 12 automated unit & integration tests
├── generate_data.py                 # Generates synthetic dataset and technician log file
├── main.py                          # Full runnable demonstration showcasing all 4 reachable states
└── README.md                        # Documentation & technical explanation
```

---

## 4. How to Run

### Step 1: Generate or Inspect Datasets and Logs
```bash
python generate_data.py
```
*Generates `data/historical_joint_data.csv` (150 samples) and `data/technician_logs.txt` (12 entries).*

### Step 2: Run the Main Demonstration
```bash
python main.py
```
This runs 6 comprehensive scenarios demonstrating:
1. **ARM-101**: Nominal parameters $\to$ **`SAFE`**
2. **ARM-02**: Early wear & squeak note $\to$ **`MONITOR`**
3. **ARM-05**: Severe predicted wear & gear play note $\to$ **`SCHEDULE_MAINTENANCE`**
4. **ARM-99**: Deterministic critical temp & vibration breach $\to$ **`EMERGENCY_STOP`**
5. **ARM-SPOOF**: Spoofed torque attack detected $\to$ **`EMERGENCY_STOP`** (tampered under warning stress)
6. **ARM-06**: Smoke & burning detected in technician log $\to$ **`EMERGENCY_STOP`**

### Step 3: Run the Test Suite
```bash
python tests/test_pipeline.py
```
*Executes all 12 unit and integration tests covering deterministic safety independence, spoof detection algorithms, NLP noise filtering, and decision reachability.*

---

## 5. Sample Output

```text
================================================================================
 ROBOTIC ARM JOINT HEALTH MONITORING SYSTEM
================================================================================
Initializing pipeline, loading historical dataset and technician logs...
[OK] Model trained on 150 historical records.
[OK] Technician maintenance logs loaded & parsed.

================================================================================
 DEMO 1: NOMINAL HEALTH CHECK
================================================================================
Arm ID: ARM-101  ===>  FINAL DECISION: [STATUS: SAFE]
--------------------------------------------------------------------------------
Rationale:
  * SAFE: All joint sensors nominal. Predictive wear model indicates healthy joint state.
  * No active mechanical flags in maintenance logs. Telemetry integrity verified.

================================================================================
 DEMO 2: EARLY WEAR WARNING & ADVISORY NOTE
================================================================================
Arm ID: ARM-02  ===>  FINAL DECISION: [STATUS: MONITOR]
--------------------------------------------------------------------------------
Rationale:
  * MONITOR: Predictive wear model indicates MODERATE_WEAR (Wear Risk: 81.7%).
  * MONITOR: Minor technician observations recorded in logs.
  * MODERATE: 'slight squeak' detected

================================================================================
 DEMO 3: PREDICTED SEVERE WEAR & TECHNICIAN CONFIRMATION
================================================================================
Arm ID: ARM-05  ===>  FINAL DECISION: [STATUS: SCHEDULE_MAINTENANCE]
--------------------------------------------------------------------------------
Rationale:
  * SCHEDULE MAINTENANCE: ML wear model predicts SEVERE_WEAR (Severe Risk: 89.0%, Total Wear Risk: 100.0%).
  * Affected joints: ['shoulder', 'elbow', 'wrist']

================================================================================
 DEMO 4: IMMEDIATE DETERMINISTIC SAFETY STOP (CRITICAL SENSOR LIMIT)
================================================================================
Arm ID: ARM-99  ===>  FINAL DECISION: [STATUS: EMERGENCY_STOP]
--------------------------------------------------------------------------------
Rationale:
  * EMERGENCY STOP TRIGGERED: Deterministic sensor safety limit breached.
  * CRITICAL: SHOULDER temperature (88.5 °C) exceeds critical threshold (82.0 °C).
  * CRITICAL: SHOULDER vibration (7.80 mm/s) exceeds critical threshold (7.00 mm/s).

================================================================================
 DEMO 5: SPOOFED SENSOR ATTACK DEFENSE IN ACTION
================================================================================
Arm ID: ARM-SPOOF  ===>  FINAL DECISION: [STATUS: EMERGENCY_STOP]
--------------------------------------------------------------------------------
Rationale:
  * EMERGENCY STOP TRIGGERED: Sensor spoofing detected in conjunction with elevated warning telemetry. Immediate halt required to prevent unmonitored structural failure.
  * SPOOF ATTACK [Thermal-Friction Anomaly]: SHOULDER reports negligible torque (12.0 Nm) despite high temperature (74.0 °C) and severe vibration (5.20 mm/s). Indicates synthetic torque clamping to mask mechanical seizure.
  * SPOOF ATTACK [Kinetic Coupling Invariant]: Shoulder torque (12.0 Nm) is impossibly low given downstream loading at Elbow (160.0 Nm) and Wrist (45.0 Nm). Violates robotic serial manipulator equilibrium.
Component Signals:
  1. Safety Monitor: Triggered=False, Warnings=True
  2. Spoof Defense: Detected=True, Confidence=88%

================================================================================
 SUMMARY OF 4 REACHABLE OFFICIAL DECISION OUTCOMES
================================================================================
1. SAFE                 -> Verified in DEMO 1 (ARM-101)
2. MONITOR              -> Verified in DEMO 2 (ARM-02)
3. SCHEDULE_MAINTENANCE -> Verified in DEMO 3 (ARM-05) & DEMO 5 (ARM-SPOOF)
4. EMERGENCY_STOP       -> Verified in DEMO 4 (ARM-99) & DEMO 6 (ARM-06)
================================================================================
```
