import os
import random
import numpy as np
import pandas as pd


def generate_synthetic_dataset(output_path="data/historical_joint_data.csv", n_samples=150):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    for _ in range(50):
        records.append({
            "shoulder_torque": float(np.clip(np.random.normal(150.0, 20.0), 90.0, 230.0)),
            "shoulder_temperature": float(np.clip(np.random.normal(42.0, 4.0), 32.0, 52.0)),
            "shoulder_vibration": float(np.clip(np.random.normal(1.4, 0.3), 0.6, 2.3)),

            "elbow_torque": float(np.clip(np.random.normal(105.0, 15.0), 65.0, 150.0)),
            "elbow_temperature": float(np.clip(np.random.normal(40.0, 3.5), 30.0, 50.0)),
            "elbow_vibration": float(np.clip(np.random.normal(1.2, 0.25), 0.5, 2.0)),

            "wrist_torque": float(np.clip(np.random.normal(32.0, 5.0), 18.0, 48.0)),
            "wrist_temperature": float(np.clip(np.random.normal(36.0, 3.0), 28.0, 45.0)),
            "wrist_vibration": float(np.clip(np.random.normal(0.9, 0.2), 0.4, 1.6)),

            "wear_level": 0,
            "failure_mode": "none",
        })

    for _ in range(50):
        degrading_joint = random.choice(["shoulder", "elbow", "wrist", "all"])

        sh_torque = np.random.normal(210.0, 25.0) if degrading_joint in ["shoulder", "all"] else np.random.normal(155.0, 18.0)
        sh_temp = np.random.normal(61.0, 4.0) if degrading_joint in ["shoulder", "all"] else np.random.normal(46.0, 4.0)
        sh_vib = np.random.normal(3.6, 0.45) if degrading_joint in ["shoulder", "all"] else np.random.normal(1.7, 0.3)

        el_torque = np.random.normal(150.0, 20.0) if degrading_joint in ["elbow", "all"] else np.random.normal(110.0, 15.0)
        el_temp = np.random.normal(59.0, 4.0) if degrading_joint in ["elbow", "all"] else np.random.normal(43.0, 3.5)
        el_vib = np.random.normal(3.4, 0.4) if degrading_joint in ["elbow", "all"] else np.random.normal(1.5, 0.25)

        wr_torque = np.random.normal(52.0, 8.0) if degrading_joint in ["wrist", "all"] else np.random.normal(34.0, 5.0)
        wr_temp = np.random.normal(54.0, 3.5) if degrading_joint in ["wrist", "all"] else np.random.normal(38.0, 3.0)
        wr_vib = np.random.normal(2.9, 0.35) if degrading_joint in ["wrist", "all"] else np.random.normal(1.1, 0.2)

        records.append({
            "shoulder_torque": float(np.clip(sh_torque, 100.0, 275.0)),
            "shoulder_temperature": float(np.clip(sh_temp, 35.0, 67.0)),
            "shoulder_vibration": float(np.clip(sh_vib, 1.0, 4.4)),

            "elbow_torque": float(np.clip(el_torque, 70.0, 195.0)),
            "elbow_temperature": float(np.clip(el_temp, 33.0, 64.0)),
            "elbow_vibration": float(np.clip(el_vib, 0.8, 3.9)),

            "wrist_torque": float(np.clip(wr_torque, 20.0, 72.0)),
            "wrist_temperature": float(np.clip(wr_temp, 30.0, 59.0)),
            "wrist_vibration": float(np.clip(wr_vib, 0.6, 3.4)),

            "wear_level": 1,
            "failure_mode": f"{degrading_joint}_early_bearing_degradation",
        })

    for _ in range(50):
        degrading_joint = random.choice(["shoulder", "elbow", "wrist", "all"])

        sh_torque = np.random.normal(290.0, 30.0) if degrading_joint in ["shoulder", "all"] else np.random.normal(180.0, 25.0)
        sh_temp = np.random.normal(73.0, 4.5) if degrading_joint in ["shoulder", "all"] else np.random.normal(52.0, 5.0)
        sh_vib = np.random.normal(5.4, 0.6) if degrading_joint in ["shoulder", "all"] else np.random.normal(2.2, 0.4)

        el_torque = np.random.normal(215.0, 25.0) if degrading_joint in ["elbow", "all"] else np.random.normal(125.0, 18.0)
        el_temp = np.random.normal(70.0, 4.0) if degrading_joint in ["elbow", "all"] else np.random.normal(48.0, 4.0)
        el_vib = np.random.normal(5.0, 0.55) if degrading_joint in ["elbow", "all"] else np.random.normal(1.9, 0.3)

        wr_torque = np.random.normal(78.0, 10.0) if degrading_joint in ["wrist", "all"] else np.random.normal(40.0, 6.0)
        wr_temp = np.random.normal(67.0, 4.0) if degrading_joint in ["wrist", "all"] else np.random.normal(42.0, 3.5)
        wr_vib = np.random.normal(4.4, 0.5) if degrading_joint in ["wrist", "all"] else np.random.normal(1.4, 0.25)

        records.append({
            "shoulder_torque": float(np.clip(sh_torque, 120.0, 355.0)),
            "shoulder_temperature": float(np.clip(sh_temp, 40.0, 81.0)),
            "shoulder_vibration": float(np.clip(sh_vib, 1.2, 6.9)),

            "elbow_torque": float(np.clip(el_torque, 80.0, 275.0)),
            "elbow_temperature": float(np.clip(el_temp, 36.0, 77.0)),
            "elbow_vibration": float(np.clip(el_vib, 1.0, 6.4)),

            "wrist_torque": float(np.clip(wr_torque, 22.0, 102.0)),
            "wrist_temperature": float(np.clip(wr_temp, 32.0, 74.0)),
            "wrist_vibration": float(np.clip(wr_vib, 0.7, 5.9)),

            "wear_level": 2,
            "failure_mode": f"{degrading_joint}_impending_gear_flaking",
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")
    return df


def generate_sample_maintenance_logs(output_path="data/technician_logs.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample_logs = [
        "[2026-10-10 07:15] [ARM-01] (Dave H.): Replaced overhead LED light fixture in work cell A. Nominal handover.",
        "[2026-10-10 09:30] [ARM-02] (Sarah M.): Cleaned coffee spill near the floor perimeter safety fence. Arm area clean.",
        "[2026-10-10 11:00] [ARM-04] (Dave H.): Shift change handover completed. Floor swept and tools organized on cart.",
        "[2026-10-11 08:20] [ARM-03] (Alex K.): Routine visual check passed. Cable guide zip ties inspected and tight.",
        "[2026-10-10 14:10] [ARM-02] (Sarah M.): Noticed a slight squeak in wrist during cycle retraction. Might need grease soon.",
        "[2026-10-11 13:45] [ARM-01] (Alex K.): Shoulder joint felt slightly warm to touch after continuous 4-hour batch run.",
        "[2026-10-11 16:30] [ARM-03] (Dave H.): Minor backlash observed on elbow axis during precision alignment test.",
        "[2026-10-12 08:15] [ARM-03] (Sarah M.): Distinct metallic grinding noise coming from elbow gearbox during high-speed moves.",
        "[2026-10-12 09:00] [ARM-05] (Dave H.): Excessive play detected in wrist joint gears, with noticeable jitter under load.",
        "[2026-10-12 09:40] [ARM-04] (Alex K.): Shoulder actuator has abnormal resistance and heavy vibration felt through the mount.",
        "[2026-10-12 10:15] [ARM-06] (Sarah M.): Immediate hazard! Smoke detected from shoulder motor casing and burning smell present! Shutting down cell!",
        "[2026-10-12 10:25] [ARM-07] (Dave H.): Extreme metal crunching and locked up elbow axis; imminent fracture risk!",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_logs) + "\n")

    print(f"Saved {len(sample_logs)} log lines to {output_path}")


if __name__ == "__main__":
    generate_synthetic_dataset()
    generate_sample_maintenance_logs()
