from typing import List, Optional
from src.config import (
    ArmDecision,
    ArmHealthEvaluation,
    ArmTelemetry,
    MaintenanceSignalResult,
    SafetyCheckResult,
    SpoofDefenseResult,
    WearPredictionResult,
)


class DecisionEngine:
    @staticmethod
    def evaluate(
        telemetry: ArmTelemetry,
        safety_result: SafetyCheckResult,
        spoof_result: SpoofDefenseResult,
        wear_result: Optional[WearPredictionResult],
        maintenance_result: MaintenanceSignalResult,
    ) -> ArmHealthEvaluation:
        rationale: List[str] = []

        if safety_result.triggered:
            rationale.append("EMERGENCY STOP TRIGGERED: Deterministic sensor safety limit breached.")
            rationale.extend(safety_result.reasons)
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.EMERGENCY_STOP,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        if maintenance_result.severity_score >= 3:
            rationale.append("EMERGENCY STOP TRIGGERED: Critical technician safety hazard detected in logs.")
            rationale.extend(maintenance_result.flagged_issues)
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.EMERGENCY_STOP,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        if spoof_result.spoof_detected and safety_result.has_warnings:
            rationale.append(
                "EMERGENCY STOP TRIGGERED: Sensor spoofing detected in conjunction with elevated warning telemetry. "
                "Immediate halt required to prevent unmonitored structural failure."
            )
            rationale.extend(spoof_result.flags)
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.EMERGENCY_STOP,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        if spoof_result.spoof_detected:
            rationale.append(
                "SCHEDULE MAINTENANCE: Sensor spoofing / telemetry anomaly detected. "
                "Sensor diagnostic and hardware validation required before continued safe operation."
            )
            rationale.extend(spoof_result.flags)
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.SCHEDULE_MAINTENANCE,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        if wear_result and (wear_result.wear_category == "SEVERE_WEAR" or wear_result.severe_probability >= 0.50):
            rationale.append(
                f"SCHEDULE MAINTENANCE: ML wear model predicts {wear_result.wear_category} "
                f"(Severe Risk: {wear_result.severe_probability * 100:.1f}%, Total Wear Risk: {wear_result.wear_probability * 100:.1f}%)."
            )
            if wear_result.degraded_joints:
                rationale.append(f"Affected joints: {[j.value for j in wear_result.degraded_joints]}")
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.SCHEDULE_MAINTENANCE,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        if maintenance_result.severity_score >= 2:
            rationale.append("SCHEDULE MAINTENANCE: Technician logs indicate serious mechanical wear/issue.")
            rationale.extend(maintenance_result.flagged_issues)
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.SCHEDULE_MAINTENANCE,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        if safety_result.has_warnings:
            rationale.append("MONITOR: Sensor telemetry is within the elevated warning band.")
            rationale.extend(safety_result.warning_details)

        if wear_result and wear_result.wear_category == "MODERATE_WEAR":
            rationale.append(
                f"MONITOR: Predictive wear model indicates {wear_result.wear_category} "
                f"(Wear Risk: {wear_result.wear_probability * 100:.1f}%)."
            )

        if maintenance_result.severity_score == 1:
            rationale.append("MONITOR: Minor technician observations recorded in logs.")
            rationale.extend(maintenance_result.flagged_issues)

        if len(rationale) > 0:
            return ArmHealthEvaluation(
                arm_id=telemetry.arm_id,
                final_decision=ArmDecision.MONITOR,
                decision_rationale=rationale,
                safety_result=safety_result,
                spoof_result=spoof_result,
                wear_result=wear_result,
                maintenance_result=maintenance_result,
                timestamp=telemetry.timestamp,
            )

        rationale.append("SAFE: All joint sensors nominal. Predictive wear model indicates healthy joint state.")
        rationale.append("No active mechanical flags in maintenance logs. Telemetry integrity verified.")
        return ArmHealthEvaluation(
            arm_id=telemetry.arm_id,
            final_decision=ArmDecision.SAFE,
            decision_rationale=rationale,
            safety_result=safety_result,
            spoof_result=spoof_result,
            wear_result=wear_result,
            maintenance_result=maintenance_result,
            timestamp=telemetry.timestamp,
        )
