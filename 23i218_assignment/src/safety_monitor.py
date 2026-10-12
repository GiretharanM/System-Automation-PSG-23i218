from typing import List
from src.config import (
    ArmTelemetry,
    JointName,
    SAFETY_THRESHOLDS,
    SafetyCheckResult,
)


class SafetyMonitor:
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or SAFETY_THRESHOLDS

    def check_safety(self, telemetry: ArmTelemetry) -> SafetyCheckResult:
        reasons: List[str] = []
        critical_joints: List[JointName] = []
        warnings: List[str] = []

        joints = [
            (JointName.SHOULDER, telemetry.shoulder),
            (JointName.ELBOW, telemetry.elbow),
            (JointName.WRIST, telemetry.wrist),
        ]

        for joint_name, reading in joints:
            limits = self.thresholds[joint_name]
            critical_found = False

            if reading.torque >= limits["torque_critical"]:
                reasons.append(
                    f"CRITICAL: {joint_name.value.upper()} torque ({reading.torque:.1f} Nm) "
                    f"exceeds critical threshold ({limits['torque_critical']:.1f} Nm)."
                )
                critical_found = True
            elif reading.torque >= limits["torque_warning"]:
                warnings.append(
                    f"WARNING: {joint_name.value.upper()} torque ({reading.torque:.1f} Nm) "
                    f"in elevated warning band (>= {limits['torque_warning']:.1f} Nm)."
                )

            if reading.temperature >= limits["temp_critical"]:
                reasons.append(
                    f"CRITICAL: {joint_name.value.upper()} temperature ({reading.temperature:.1f} °C) "
                    f"exceeds critical threshold ({limits['temp_critical']:.1f} °C)."
                )
                critical_found = True
            elif reading.temperature >= limits["temp_warning"]:
                warnings.append(
                    f"WARNING: {joint_name.value.upper()} temperature ({reading.temperature:.1f} °C) "
                    f"in elevated warning band (>= {limits['temp_warning']:.1f} °C)."
                )

            if reading.vibration >= limits["vibration_critical"]:
                reasons.append(
                    f"CRITICAL: {joint_name.value.upper()} vibration ({reading.vibration:.2f} mm/s) "
                    f"exceeds critical threshold ({limits['vibration_critical']:.2f} mm/s)."
                )
                critical_found = True
            elif reading.vibration >= limits["vibration_warning"]:
                warnings.append(
                    f"WARNING: {joint_name.value.upper()} vibration ({reading.vibration:.2f} mm/s) "
                    f"in elevated warning band (>= {limits['vibration_warning']:.2f} mm/s)."
                )

            if critical_found and joint_name not in critical_joints:
                critical_joints.append(joint_name)

        return SafetyCheckResult(
            triggered=len(reasons) > 0,
            reasons=reasons,
            critical_joints=critical_joints,
            has_warnings=len(warnings) > 0,
            warning_details=warnings,
        )
