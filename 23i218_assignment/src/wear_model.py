import os
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.config import (
    ArmTelemetry,
    JointName,
    WearPredictionResult,
)


WEAR_LABELS = {
    0: "HEALTHY",
    1: "MODERATE_WEAR",
    2: "SEVERE_WEAR",
}


class PredictiveWearModel:
    def __init__(self, model: Optional[RandomForestClassifier] = None):
        self.model = model or RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        )
        self.is_trained = False
        self.feature_names = ArmTelemetry.FEATURE_NAMES

    def train_from_csv(self, csv_path: str) -> dict:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Historical dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = self.feature_names + ["wear_level"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in dataset: {col}")

        X = df[self.feature_names].values
        y = df["wear_level"].values

        self.model.fit(X, y)
        self.is_trained = True

        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)

        return {
            "samples": len(df),
            "accuracy": float(acc),
            "feature_importances": dict(
                zip(self.feature_names, [float(x) for x in self.model.feature_importances_])
            ),
        }

    def predict(self, telemetry: ArmTelemetry) -> WearPredictionResult:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train_from_csv first.")

        x = np.array(telemetry.to_feature_vector()).reshape(1, -1)
        pred_class = int(self.model.predict(x)[0])
        probabilities = self.model.predict_proba(x)[0]

        classes = list(self.model.classes_)
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}

        wear_category = WEAR_LABELS.get(pred_class, "UNKNOWN")
        prob_wear = float(prob_dict.get(1, 0.0) + prob_dict.get(2, 0.0))
        prob_severe = float(prob_dict.get(2, 0.0))

        degraded_joints = self._identify_degraded_joints(telemetry)

        return WearPredictionResult(
            wear_category=wear_category,
            wear_probability=round(prob_wear, 3),
            severe_probability=round(prob_severe, 3),
            degraded_joints=degraded_joints,
            raw_class=pred_class,
        )

    def _identify_degraded_joints(self, telemetry: ArmTelemetry) -> List[JointName]:
        degraded = []
        if telemetry.shoulder.vibration > 3.0 or telemetry.shoulder.temperature > 58.0:
            degraded.append(JointName.SHOULDER)
        if telemetry.elbow.vibration > 2.8 or telemetry.elbow.temperature > 55.0:
            degraded.append(JointName.ELBOW)
        if telemetry.wrist.vibration > 2.2 or telemetry.wrist.temperature > 52.0:
            degraded.append(JointName.WRIST)
        return degraded
