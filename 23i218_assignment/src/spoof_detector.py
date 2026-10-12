from typing import List, Optional
from src.config import (
    ArmTelemetry,
    JointName,
    PHYSICAL_BOUNDS,
    SpoofDefenseResult,
)


class SpoofDetector:
    def __init__(self, bounds: dict = None):
        self.bounds = bounds or PHYSICAL_BOUNDS

    def evaluate(
        self,
        telemetry: ArmTelemetry,
        history: Optional[List[ArmTelemetry]] = None,
    ) -> SpoofDefenseResult:
        flags: List[str] = []
        suspect_joints: List[JointName] = []
        confidence_points: float = 0.0

        joints = [
            (JointName.SHOULDER, telemetry.shoulder),
            (JointName.ELBOW, telemetry.elbow),
            (JointName.WRIST, telemetry.wrist),
        ]

        for joint_name, reading in joints:
            if reading.temperature >= 60.0 and reading.vibration >= 3.5:
                min_expected_torque = 25.0 if joint_name != JointName.WRIST else 8.0
                if reading.torque < min_expected_torque:
                    flags.append(
                        f"SPOOF ATTACK [Thermal-Friction Anomaly]: {joint_name.value.upper()} "
                        f"reports negligible torque ({reading.torque:.1f} Nm) despite high temperature "
                        f"({reading.temperature:.1f} °C) and severe vibration ({reading.vibration:.2f} mm/s). "
                        "Indicates synthetic torque clamping to mask mechanical seizure."
                    )
                    suspect_joints.append(joint_name)
                    confidence_points += 0.85

        shoulder_torque = telemetry.shoulder.torque
        elbow_torque = telemetry.elbow.torque
        wrist_torque = telemetry.wrist.torque

        if (elbow_torque > 110.0 or wrist_torque > 35.0) and shoulder_torque < 30.0:
            flags.append(
                f"SPOOF ATTACK [Kinetic Coupling Invariant]: Shoulder torque ({shoulder_torque:.1f} Nm) "
                f"is impossibly low given downstream loading at Elbow ({elbow_torque:.1f} Nm) "
                f"and Wrist ({wrist_torque:.1f} Nm). Violates robotic serial manipulator equilibrium."
            )
            if JointName.SHOULDER not in suspect_joints:
                suspect_joints.append(JointName.SHOULDER)
            confidence_points += 0.90

        for joint_name, reading in joints:
            b = self.bounds[joint_name]
            if reading.torque < b["min_torque"] or reading.torque > b["max_torque"]:
                flags.append(
                    f"SPOOF ATTACK [Boundary Violation]: {joint_name.value.upper()} torque "
                    f"({reading.torque:.1f} Nm) outside physical domain [{b['min_torque']}, {b['max_torque']}]."
                )
                if joint_name not in suspect_joints:
                    suspect_joints.append(joint_name)
                confidence_points += 0.95

            if reading.temperature < b["min_temp"] or reading.temperature > b["max_temp"]:
                flags.append(
                    f"SPOOF ATTACK [Boundary Violation]: {joint_name.value.upper()} temperature "
                    f"({reading.temperature:.1f} °C) outside plausible facility ambient domain."
                )
                if joint_name not in suspect_joints:
                    suspect_joints.append(joint_name)
                confidence_points += 0.95

        if history and len(history) >= 4:
            for joint_name in [JointName.SHOULDER, JointName.ELBOW, JointName.WRIST]:
                torques = [h.get_joint(joint_name).torque for h in history] + [
                    telemetry.get_joint(joint_name).torque
                ]
                if len(set(torques)) == 1 and torques[0] > 0:
                    flags.append(
                        f"SPOOF ATTACK [Frozen Signal]: {joint_name.value.upper()} torque signal "
                        f"is frozen at exactly {torques[0]:.4f} Nm across multiple samples without natural ADC jitter."
                    )
                    if joint_name not in suspect_joints:
                        suspect_joints.append(joint_name)
                    confidence_points += 0.75

        spoof_detected = len(flags) > 0
        normalized_confidence = min(1.0, max(0.0, confidence_points / max(1, len(flags)))) if spoof_detected else 0.0

        return SpoofDefenseResult(
            spoof_detected=spoof_detected,
            confidence=round(normalized_confidence, 2),
            flags=flags,
            suspect_joints=list(set(suspect_joints)),
        )
