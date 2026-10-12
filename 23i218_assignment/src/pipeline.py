from typing import Dict, List, Optional
from src.config import (
    ArmHealthEvaluation,
    ArmTelemetry,
)
from src.safety_monitor import SafetyMonitor
from src.spoof_detector import SpoofDetector
from src.wear_model import PredictiveWearModel
from src.maintenance_nlp import MaintenanceLogParser
from src.decision_engine import DecisionEngine


class JointHealthPipeline:
    def __init__(
        self,
        dataset_path: Optional[str] = "data/historical_joint_data.csv",
        logs_path: Optional[str] = "data/technician_logs.txt",
    ):
        self.safety_monitor = SafetyMonitor()
        self.spoof_detector = SpoofDetector()
        self.wear_model = PredictiveWearModel()
        self.log_parser = MaintenanceLogParser(logs_path)
        self.telemetry_history: Dict[str, List[ArmTelemetry]] = {}

        if dataset_path:
            self.wear_model.train_from_csv(dataset_path)

    def process_arm(
        self,
        telemetry: ArmTelemetry,
        direct_notes: Optional[List[str]] = None,
    ) -> ArmHealthEvaluation:
        safety_result = self.safety_monitor.check_safety(telemetry)

        history = self.telemetry_history.get(telemetry.arm_id, [])
        spoof_result = self.spoof_detector.evaluate(telemetry, history=history)

        if telemetry.arm_id not in self.telemetry_history:
            self.telemetry_history[telemetry.arm_id] = []
        self.telemetry_history[telemetry.arm_id].append(telemetry)
        if len(self.telemetry_history[telemetry.arm_id]) > 10:
            self.telemetry_history[telemetry.arm_id].pop(0)

        wear_result = None
        if self.wear_model.is_trained:
            wear_result = self.wear_model.predict(telemetry)

        maintenance_result = self.log_parser.analyze_arm_notes(
            arm_id=telemetry.arm_id,
            direct_notes=direct_notes,
        )

        return DecisionEngine.evaluate(
            telemetry=telemetry,
            safety_result=safety_result,
            spoof_result=spoof_result,
            wear_result=wear_result,
            maintenance_result=maintenance_result,
        )
