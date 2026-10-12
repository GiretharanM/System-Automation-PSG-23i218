import re
from typing import Dict, List, Optional
from src.config import JointName, MaintenanceSignalResult


CRITICAL_KEYWORDS = [
    "smoke", "burning", "burnt", "sparks", "fire", "fracture", "cracked gear",
    "seizure", "locked up", "catastrophic", "imminent failure", "severe crunching",
    "danger", "emergency", "shut down immediately", "arcing", "screeching halt",
]

HIGH_SEVERITY_KEYWORDS = [
    "grinding", "heavy wear", "hot to touch", "overheating", "excessive play",
    "clunking", "binding", "severe vibration", "backlash", "slipping",
    "gear teeth worn", "loose mounting", "shaking violently", "stiff joint",
    "fluid leak", "bearing rumble", "chatter", "abnormal resistance",
]

MODERATE_SEVERITY_KEYWORDS = [
    "slight squeak", "faint grinding", "warm to touch", "minor play", "jitter",
    "intermittent click", "low lubrication", "needs grease", "dry seal",
    "unusual hum", "whine", "hesitation", "minor wobble", "rough motion",
]

NOISE_INDICATORS = [
    "coffee", "lunch", "handover", "shift change", "light tube", "flooring",
    "safety fence", "zip tie", "trash", "whiteboard", "safety glasses",
    "routine check passed", "all good", "nominal", "clean", "swept",
]


class MaintenanceLogParser:
    def __init__(self, log_file_path: Optional[str] = None):
        self.log_file_path = log_file_path
        self.entries: List[Dict[str, str]] = []
        if log_file_path:
            self.load_log_file(log_file_path)

    def load_log_file(self, path: str):
        self.entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parsed = self.parse_single_line(line)
                if parsed:
                    self.entries.append(parsed)

    @staticmethod
    def parse_single_line(raw_line: str) -> Optional[Dict[str, str]]:
        match = re.match(
            r"^\[(?P<timestamp>[^\]]+)\]\s+\[(?P<arm_id>[^\]]+)\](?:\s*\((?P<author>[^\)]+)\))?:\s*(?P<message>.+)$",
            raw_line,
            re.IGNORECASE,
        )
        if match:
            return {
                "timestamp": match.group("timestamp").strip(),
                "arm_id": match.group("arm_id").strip().upper(),
                "author": match.group("author").strip() if match.group("author") else "Tech",
                "message": match.group("message").strip(),
            }

        arm_match = re.search(r"\b(ARM[-_]?\d+|ROBOT[-_]?[A-Z0-9]+)\b", raw_line, re.IGNORECASE)
        if arm_match:
            arm_id = arm_match.group(1).upper().replace("_", "-")
            return {
                "timestamp": "UNKNOWN",
                "arm_id": arm_id,
                "author": "Tech",
                "message": raw_line,
            }

        return None

    def analyze_arm_notes(self, arm_id: str, direct_notes: Optional[List[str]] = None) -> MaintenanceSignalResult:
        target_id = arm_id.upper().replace("_", "-")
        matched_notes = []

        for entry in self.entries:
            entry_arm = entry["arm_id"].replace("_", "-")
            if entry_arm == target_id:
                matched_notes.append(entry["message"])

        if direct_notes:
            matched_notes.extend(direct_notes)

        if not matched_notes:
            return MaintenanceSignalResult(
                entries_found=0,
                severity_score=0,
                flagged_issues=[],
                mentioned_joints=[],
                relevant_notes=[],
            )

        max_severity = 0
        flagged_issues: List[str] = []
        mentioned_joints: List[JointName] = []
        relevant_notes: List[str] = []

        for note in matched_notes:
            note_lower = note.lower()

            is_noise = any(indicator in note_lower for indicator in NOISE_INDICATORS)
            has_joint_mention = any(j.value in note_lower for j in [JointName.SHOULDER, JointName.ELBOW, JointName.WRIST])

            current_joints = []
            if "shoulder" in note_lower:
                current_joints.append(JointName.SHOULDER)
            if "elbow" in note_lower:
                current_joints.append(JointName.ELBOW)
            if "wrist" in note_lower:
                current_joints.append(JointName.WRIST)

            note_severity = 0
            found_reasons = []

            for kw in CRITICAL_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", note_lower):
                    note_severity = max(note_severity, 3)
                    found_reasons.append(f"CRITICAL: '{kw}' detected")

            for kw in HIGH_SEVERITY_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", note_lower):
                    note_severity = max(note_severity, 2)
                    found_reasons.append(f"HIGH: '{kw}' detected")

            for kw in MODERATE_SEVERITY_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", note_lower):
                    note_severity = max(note_severity, 1)
                    found_reasons.append(f"MODERATE: '{kw}' detected")

            if note_severity > 0 or (has_joint_mention and not is_noise):
                relevant_notes.append(note)
                max_severity = max(max_severity, note_severity)
                flagged_issues.extend(found_reasons)
                for j in current_joints:
                    if j not in mentioned_joints:
                        mentioned_joints.append(j)

        return MaintenanceSignalResult(
            entries_found=len(matched_notes),
            severity_score=max_severity,
            flagged_issues=list(set(flagged_issues)),
            mentioned_joints=mentioned_joints,
            relevant_notes=relevant_notes,
        )
