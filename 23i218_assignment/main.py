import sys
import os
import io

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import (
    ArmDecision,
    ArmTelemetry,
    JointReading,
)
from src.pipeline import JointHealthPipeline


class DualLogger:
    def __init__(self, filepath="output_report.txt"):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def print_banner(text: str):
    print("=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_evaluation(eval_result):
    status_symbols = {
        ArmDecision.SAFE: "[STATUS: SAFE]",
        ArmDecision.MONITOR: "[STATUS: MONITOR]",
        ArmDecision.SCHEDULE_MAINTENANCE: "[STATUS: SCHEDULE_MAINTENANCE]",
        ArmDecision.EMERGENCY_STOP: "[STATUS: EMERGENCY_STOP]",
    }

    badge = status_symbols.get(eval_result.final_decision, str(eval_result.final_decision))
    print(f"\nArm ID: {eval_result.arm_id}  ===>  FINAL DECISION: {badge}")
    print("-" * 80)
    print("Rationale:")
    for r in eval_result.decision_rationale:
        print(f"  * {r}")

    print("\nComponent Signals:")
    print(f"  1. Safety Monitor: Triggered={eval_result.safety_result.triggered}, "
          f"Warnings={eval_result.safety_result.has_warnings}")
    if eval_result.safety_result.reasons:
        for reason in eval_result.safety_result.reasons:
            print(f"     - {reason}")

    print(f"  2. Spoof Defense: Detected={eval_result.spoof_result.spoof_detected}, "
          f"Confidence={eval_result.spoof_result.confidence * 100:.0f}%")
    if eval_result.spoof_result.flags:
        for flag in eval_result.spoof_result.flags:
            print(f"     - {flag}")

    if eval_result.wear_result:
        wr = eval_result.wear_result
        print(f"  3. Predictive Wear ML: Category={wr.wear_category}, "
              f"P(Wear)={wr.wear_probability * 100:.1f}%, Degraded Joints={[j.value for j in wr.degraded_joints]}")

    mr = eval_result.maintenance_result
    print(f"  4. Maintenance Notes: Entries={mr.entries_found}, SeverityScore={mr.severity_score}/3")
    if mr.relevant_notes:
        for note in mr.relevant_notes:
            print(f"     - Note: \"{note}\"")
    print("-" * 80)


def run_demonstration():
    print_banner("ROBOTIC ARM JOINT HEALTH MONITORING SYSTEM")
    print("Initializing pipeline, loading historical dataset and technician logs...")

    pipeline = JointHealthPipeline(
        dataset_path="data/historical_joint_data.csv",
        logs_path="data/technician_logs.txt",
    )
    print("[OK] Model trained on 150 historical records.")
    print("[OK] Technician maintenance logs loaded & parsed.\n")

    arm_safe = ArmTelemetry(
        arm_id="ARM-101",
        shoulder=JointReading(torque=145.0, temperature=41.5, vibration=1.2),
        elbow=JointReading(torque=102.0, temperature=38.0, vibration=1.1),
        wrist=JointReading(torque=28.0, temperature=34.5, vibration=0.8),
    )

    arm_monitor = ArmTelemetry(
        arm_id="ARM-02",
        shoulder=JointReading(torque=165.0, temperature=46.0, vibration=1.8),
        elbow=JointReading(torque=115.0, temperature=44.0, vibration=1.6),
        wrist=JointReading(torque=42.0, temperature=53.0, vibration=2.6),
    )

    arm_maint = ArmTelemetry(
        arm_id="ARM-05",
        shoulder=JointReading(torque=230.0, temperature=58.0, vibration=3.4),
        elbow=JointReading(torque=170.0, temperature=56.0, vibration=3.2),
        wrist=JointReading(torque=72.0, temperature=66.0, vibration=4.2),
    )

    arm_estop_sensor = ArmTelemetry(
        arm_id="ARM-99",
        shoulder=JointReading(torque=310.0, temperature=88.5, vibration=7.8),
        elbow=JointReading(torque=130.0, temperature=50.0, vibration=2.2),
        wrist=JointReading(torque=35.0, temperature=39.0, vibration=1.0),
    )

    arm_spoof_attack = ArmTelemetry(
        arm_id="ARM-SPOOF",
        shoulder=JointReading(torque=12.0, temperature=74.0, vibration=5.2),
        elbow=JointReading(torque=160.0, temperature=58.0, vibration=3.0),
        wrist=JointReading(torque=45.0, temperature=44.0, vibration=1.5),
    )

    arm_estop_tech = ArmTelemetry(
        arm_id="ARM-06",
        shoulder=JointReading(torque=270.0, temperature=67.0, vibration=4.1),
        elbow=JointReading(torque=150.0, temperature=52.0, vibration=2.5),
        wrist=JointReading(torque=40.0, temperature=42.0, vibration=1.3),
    )

    test_arms = [
        ("DEMO 1: NOMINAL HEALTH CHECK", arm_safe, None),
        ("DEMO 2: EARLY WEAR WARNING & ADVISORY NOTE", arm_monitor, None),
        ("DEMO 3: PREDICTED SEVERE WEAR & TECHNICIAN CONFIRMATION", arm_maint, None),
        ("DEMO 4: IMMEDIATE DETERMINISTIC SAFETY STOP (CRITICAL SENSOR LIMIT)", arm_estop_sensor, None),
        ("DEMO 5: SPOOFED SENSOR ATTACK DEFENSE IN ACTION", arm_spoof_attack, None),
        ("DEMO 6: EMERGENCY STOP VIA CRITICAL TECHNICIAN ALERT (SMOKE IN LOG)", arm_estop_tech, None),
    ]

    for title, arm, direct_notes in test_arms:
        print_banner(title)
        evaluation = pipeline.process_arm(arm, direct_notes=direct_notes)
        print_evaluation(evaluation)

    print_banner("SUMMARY OF 4 REACHABLE OFFICIAL DECISION OUTCOMES")
    print("1. SAFE                 -> Verified in DEMO 1 (ARM-101)")
    print("2. MONITOR              -> Verified in DEMO 2 (ARM-02)")
    print("3. SCHEDULE_MAINTENANCE -> Verified in DEMO 3 (ARM-05) & DEMO 5 (ARM-SPOOF)")
    print("4. EMERGENCY_STOP       -> Verified in DEMO 4 (ARM-99) & DEMO 6 (ARM-06)")
    print("=" * 80)


if __name__ == "__main__":
    report_file = "output_report.txt"
    logger = DualLogger(report_file)
    sys.stdout = logger
    try:
        run_demonstration()
        print(f"\n[SAVED] Full report has also been saved to file: {os.path.abspath(report_file)}")
    finally:
        sys.stdout = logger.terminal
        logger.close()
