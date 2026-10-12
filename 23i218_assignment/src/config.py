from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ArmDecision(str, Enum):
    SAFE = "SAFE"
    MONITOR = "MONITOR"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class JointName(str, Enum):
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"


SAFETY_THRESHOLDS = {
    JointName.SHOULDER: {
        "torque_warning": 280.0,
        "torque_critical": 360.0,
        "temp_warning": 68.0,
        "temp_critical": 82.0,
        "vibration_warning": 4.5,
        "vibration_critical": 7.0,
    },
    JointName.ELBOW: {
        "torque_warning": 200.0,
        "torque_critical": 280.0,
        "temp_warning": 65.0,
        "temp_critical": 78.0,
        "vibration_warning": 4.0,
        "vibration_critical": 6.5,
    },
    JointName.WRIST: {
        "torque_warning": 75.0,
        "torque_critical": 105.0,
        "temp_warning": 60.0,
        "temp_critical": 75.0,
        "vibration_warning": 3.5,
        "vibration_critical": 6.0,
    },
}

PHYSICAL_BOUNDS = {
    JointName.SHOULDER: {"min_torque": 0.0, "max_torque": 500.0, "min_temp": -10.0, "max_temp": 150.0, "min_vib": 0.0},
    JointName.ELBOW:    {"min_torque": 0.0, "max_torque": 400.0, "min_temp": -10.0, "max_temp": 150.0, "min_vib": 0.0},
    JointName.WRIST:    {"min_torque": 0.0, "max_torque": 200.0, "min_temp": -10.0, "max_temp": 150.0, "min_vib": 0.0},
}


@dataclass
class JointReading:
    torque: float
    temperature: float
    vibration: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "torque": self.torque,
            "temperature": self.temperature,
            "vibration": self.vibration,
        }


@dataclass
class ArmTelemetry:
    arm_id: str
    shoulder: JointReading
    elbow: JointReading
    wrist: JointReading
    timestamp: Optional[str] = None

    def get_joint(self, joint: JointName) -> JointReading:
        if joint == JointName.SHOULDER:
            return self.shoulder
        elif joint == JointName.ELBOW:
            return self.elbow
        elif joint == JointName.WRIST:
            return self.wrist
        raise ValueError(f"Unknown joint: {joint}")

    def to_feature_vector(self) -> List[float]:
        return [
            self.shoulder.torque,
            self.shoulder.temperature,
            self.shoulder.vibration,
            self.elbow.torque,
            self.elbow.temperature,
            self.elbow.vibration,
            self.wrist.torque,
            self.wrist.temperature,
            self.wrist.vibration,
        ]

    FEATURE_NAMES = [
        "shoulder_torque", "shoulder_temperature", "shoulder_vibration",
        "elbow_torque", "elbow_temperature", "elbow_vibration",
        "wrist_torque", "wrist_temperature", "wrist_vibration",
    ]


@dataclass
class SafetyCheckResult:
    triggered: bool
    reasons: List[str] = field(default_factory=list)
    critical_joints: List[JointName] = field(default_factory=list)
    has_warnings: bool = False
    warning_details: List[str] = field(default_factory=list)


@dataclass
class SpoofDefenseResult:
    spoof_detected: bool
    confidence: float
    flags: List[str] = field(default_factory=list)
    suspect_joints: List[JointName] = field(default_factory=list)


@dataclass
class WearPredictionResult:
    wear_category: str
    wear_probability: float
    severe_probability: float
    degraded_joints: List[JointName] = field(default_factory=list)
    raw_class: int = 0


@dataclass
class MaintenanceSignalResult:
    entries_found: int = 0
    severity_score: int = 0
    flagged_issues: List[str] = field(default_factory=list)
    mentioned_joints: List[JointName] = field(default_factory=list)
    relevant_notes: List[str] = field(default_factory=list)


@dataclass
class ArmHealthEvaluation:
    arm_id: str
    final_decision: ArmDecision
    decision_rationale: List[str]
    safety_result: SafetyCheckResult
    spoof_result: SpoofDefenseResult
    wear_result: Optional[WearPredictionResult]
    maintenance_result: MaintenanceSignalResult
    timestamp: Optional[str] = None
