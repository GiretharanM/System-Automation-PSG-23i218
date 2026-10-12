import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    ArmDecision,
    ArmTelemetry,
    JointName,
    JointReading,
)
from src.safety_monitor import SafetyMonitor
from src.spoof_detector import SpoofDetector
from src.maintenance_nlp import MaintenanceLogParser
from src.decision_engine import DecisionEngine
from src.pipeline import JointHealthPipeline


class TestRoboticArmMonitoringSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = JointHealthPipeline(
            dataset_path="data/historical_joint_data.csv",
            logs_path="data/technician_logs.txt",
        )

    def test_immediate_safety_stop_critical_torque(self):
        monitor = SafetyMonitor()
        telemetry = ArmTelemetry(
            arm_id="TEST-CRIT-TORQUE",
            shoulder=JointReading(torque=375.0, temperature=45.0, vibration=1.5),
            elbow=JointReading(torque=100.0, temperature=40.0, vibration=1.0),
            wrist=JointReading(torque=30.0, temperature=35.0, vibration=0.8),
        )
        res = monitor.check_safety(telemetry)
        self.assertTrue(res.triggered)
        self.assertIn(JointName.SHOULDER, res.critical_joints)
        self.assertTrue(any("torque" in r.lower() for r in res.reasons))

    def test_immediate_safety_stop_critical_temp(self):
        monitor = SafetyMonitor()
        telemetry = ArmTelemetry(
            arm_id="TEST-CRIT-TEMP",
            shoulder=JointReading(torque=150.0, temperature=45.0, vibration=1.5),
            elbow=JointReading(torque=100.0, temperature=80.5, vibration=1.0),
            wrist=JointReading(torque=30.0, temperature=35.0, vibration=0.8),
        )
        res = monitor.check_safety(telemetry)
        self.assertTrue(res.triggered)
        self.assertIn(JointName.ELBOW, res.critical_joints)

    def test_immediate_safety_stop_critical_vibration(self):
        monitor = SafetyMonitor()
        telemetry = ArmTelemetry(
            arm_id="TEST-CRIT-VIB",
            shoulder=JointReading(torque=150.0, temperature=45.0, vibration=1.5),
            elbow=JointReading(torque=100.0, temperature=40.0, vibration=1.0),
            wrist=JointReading(torque=30.0, temperature=35.0, vibration=6.8),
        )
        res = monitor.check_safety(telemetry)
        self.assertTrue(res.triggered)
        self.assertIn(JointName.WRIST, res.critical_joints)

    def test_safety_stop_independent_of_ml(self):
        telemetry = ArmTelemetry(
            arm_id="TEST-NO-ML",
            shoulder=JointReading(torque=390.0, temperature=50.0, vibration=2.0),
            elbow=JointReading(torque=100.0, temperature=40.0, vibration=1.0),
            wrist=JointReading(torque=30.0, temperature=35.0, vibration=0.8),
        )
        safety_res = self.pipeline.safety_monitor.check_safety(telemetry)
        spoof_res = self.pipeline.spoof_detector.evaluate(telemetry)
        maint_res = self.pipeline.log_parser.analyze_arm_notes(telemetry.arm_id)

        eval_result = DecisionEngine.evaluate(
            telemetry=telemetry,
            safety_result=safety_res,
            spoof_result=spoof_res,
            wear_result=None,
            maintenance_result=maint_res,
        )
        self.assertEqual(eval_result.final_decision, ArmDecision.EMERGENCY_STOP)

    def test_predictive_wear_model_accuracy(self):
        self.assertTrue(self.pipeline.wear_model.is_trained)
        healthy_telemetry = ArmTelemetry(
            arm_id="TEST-WEAR-HEALTHY",
            shoulder=JointReading(torque=140.0, temperature=40.0, vibration=1.1),
            elbow=JointReading(torque=95.0, temperature=38.0, vibration=1.0),
            wrist=JointReading(torque=28.0, temperature=34.0, vibration=0.7),
        )
        pred = self.pipeline.wear_model.predict(healthy_telemetry)
        self.assertEqual(pred.wear_category, "HEALTHY")

    def test_maintenance_nlp_noise_rejection(self):
        parser = MaintenanceLogParser()
        res = parser.analyze_arm_notes("ARM-TEST", direct_notes=[
            "Replaced coffee machine filter near assembly cell.",
            "Floor swept and housekeeping complete.",
        ])
        self.assertEqual(res.severity_score, 0)
        self.assertEqual(len(res.relevant_notes), 0)

    def test_maintenance_nlp_joint_and_severity_detection(self):
        parser = MaintenanceLogParser()
        res = parser.analyze_arm_notes("ARM-TEST", direct_notes=[
            "Technician noticed metallic grinding noise in elbow gearbox.",
        ])
        self.assertEqual(res.severity_score, 2)
        self.assertIn(JointName.ELBOW, res.mentioned_joints)

    def test_maintenance_nlp_critical_hazard(self):
        parser = MaintenanceLogParser()
        res = parser.analyze_arm_notes("ARM-TEST", direct_notes=[
            "Smoke coming from shoulder joint! Immediate shutdown required!",
        ])
        self.assertEqual(res.severity_score, 3)
        self.assertIn(JointName.SHOULDER, res.mentioned_joints)

    def test_spoof_defense_thermal_friction_contradiction(self):
        detector = SpoofDetector()
        telemetry = ArmTelemetry(
            arm_id="TEST-SPOOF-THERMAL",
            shoulder=JointReading(torque=10.0, temperature=75.0, vibration=5.0),
            elbow=JointReading(torque=100.0, temperature=40.0, vibration=1.0),
            wrist=JointReading(torque=30.0, temperature=35.0, vibration=0.8),
        )
        res = detector.evaluate(telemetry)
        self.assertTrue(res.spoof_detected)
        self.assertIn(JointName.SHOULDER, res.suspect_joints)
        self.assertTrue(any("Thermal-Friction" in f for f in res.flags))

    def test_spoof_defense_kinetic_coupling_contradiction(self):
        detector = SpoofDetector()
        telemetry = ArmTelemetry(
            arm_id="TEST-SPOOF-KINETIC",
            shoulder=JointReading(torque=5.0, temperature=42.0, vibration=1.2),
            elbow=JointReading(torque=175.0, temperature=48.0, vibration=1.8),
            wrist=JointReading(torque=45.0, temperature=38.0, vibration=1.0),
        )
        res = detector.evaluate(telemetry)
        self.assertTrue(res.spoof_detected)
        self.assertIn(JointName.SHOULDER, res.suspect_joints)
        self.assertTrue(any("Kinetic Coupling" in f for f in res.flags))

    def test_spoof_defense_physical_bounds(self):
        detector = SpoofDetector()
        telemetry = ArmTelemetry(
            arm_id="TEST-SPOOF-BOUNDS",
            shoulder=JointReading(torque=-25.0, temperature=42.0, vibration=1.2),
            elbow=JointReading(torque=100.0, temperature=40.0, vibration=1.0),
            wrist=JointReading(torque=30.0, temperature=35.0, vibration=0.8),
        )
        res = detector.evaluate(telemetry)
        self.assertTrue(res.spoof_detected)
        self.assertTrue(any("Boundary Violation" in f for f in res.flags))

    def test_all_four_decisions_reachable(self):
        arm_safe = ArmTelemetry(
            arm_id="ARM-101",
            shoulder=JointReading(torque=140.0, temperature=40.0, vibration=1.1),
            elbow=JointReading(torque=95.0, temperature=38.0, vibration=1.0),
            wrist=JointReading(torque=28.0, temperature=34.0, vibration=0.7),
        )
        res_safe = self.pipeline.process_arm(arm_safe)
        self.assertEqual(res_safe.final_decision, ArmDecision.SAFE)

        arm_mon = ArmTelemetry(
            arm_id="ARM-02",
            shoulder=JointReading(torque=160.0, temperature=46.0, vibration=1.8),
            elbow=JointReading(torque=115.0, temperature=44.0, vibration=1.6),
            wrist=JointReading(torque=42.0, temperature=53.0, vibration=2.6),
        )
        res_mon = self.pipeline.process_arm(arm_mon)
        self.assertEqual(res_mon.final_decision, ArmDecision.MONITOR)

        arm_maint = ArmTelemetry(
            arm_id="ARM-05",
            shoulder=JointReading(torque=230.0, temperature=58.0, vibration=3.4),
            elbow=JointReading(torque=170.0, temperature=56.0, vibration=3.2),
            wrist=JointReading(torque=72.0, temperature=66.0, vibration=4.2),
        )
        res_maint = self.pipeline.process_arm(arm_maint)
        self.assertEqual(res_maint.final_decision, ArmDecision.SCHEDULE_MAINTENANCE)

        arm_estop = ArmTelemetry(
            arm_id="ARM-99",
            shoulder=JointReading(torque=380.0, temperature=85.0, vibration=7.5),
            elbow=JointReading(torque=110.0, temperature=42.0, vibration=1.2),
            wrist=JointReading(torque=32.0, temperature=35.0, vibration=0.8),
        )
        res_estop = self.pipeline.process_arm(arm_estop)
        self.assertEqual(res_estop.final_decision, ArmDecision.EMERGENCY_STOP)


if __name__ == "__main__":
    unittest.main()
