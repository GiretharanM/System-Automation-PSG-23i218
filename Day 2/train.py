
import ast
import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# ==========================================
# STEP 1: FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "m_s_d.csv"
LOG_FILE = BASE_DIR.parent / "day1_log.txt"
OUTPUT_FILE = BASE_DIR / "log_ml_comparison.csv"

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "current"
]

VALID_STATES = ["NORMAL", "WARNING", "UNSAFE"]


# ==========================================
# STEP 2: CONVERT ACTION TO STATE
# ==========================================

def action_to_state(action):
    action = str(action).strip().upper()

    mapping = {
        "CONTINUE_NORMAL": "NORMAL",
        "CURRENT_WARNING": "WARNING",
        "PRESSURE_WARNING": "WARNING",
        "VIBRATION_WARNING": "WARNING",
        "STOP_MACHINE": "UNSAFE"
    }

    return mapping.get(action, "UNKNOWN")


# ==========================================
# STEP 3: LOAD TRAINING DATA
# ==========================================

def load_training_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
    )

    # Support alternate sensor column names
    aliases = {
        "temperature": ["temp"],
        "pressure": [],
        "vibration": [],
        "current": ["motor_current"]
    }

    for target, alternatives in aliases.items():
        if target not in df.columns:
            for alternative in alternatives:
                if alternative in df.columns:
                    df.rename(
                        columns={alternative: target},
                        inplace=True
                    )
                    break

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing sensor columns in m_s_d.csv: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )

    # Use existing state labels if available.
    # Otherwise derive them from action or rule.
    if "state" in df.columns:
        df["state"] = (
            df["state"].astype(str).str.strip().str.upper()
        )

    elif "action" in df.columns:
        df["state"] = df["action"].apply(action_to_state)

    elif "rule" in df.columns:
        rule_mapping = {
            "NO_RULE_FIRED": "NORMAL",
            "HIGH_CURRENT": "WARNING",
            "HIGH_PRESSURE": "WARNING",
            "HIGH_VIBRATION": "WARNING",
            "EXCESSIVE_VIBRATION": "WARNING",
            "HIGH_TEMP": "UNSAFE",
            "HIGH_TEMPERATURE": "UNSAFE",
            "CRITICAL_TEMPERATURE": "UNSAFE"
        }

        df["state"] = (
            df["rule"].astype(str).str.strip().str.upper()
            .map(rule_mapping)
            .fillna("UNKNOWN")
        )

    else:
        raise ValueError(
            "CSV needs a 'state', 'action', or 'rule' column."
        )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    df.replace([float("inf"), float("-inf")], pd.NA,
               inplace=True)

    df.dropna(subset=FEATURES + ["state"], inplace=True)
    df = df[df["state"].isin(VALID_STATES)].copy()

    if len(df) < 5:
        raise ValueError(
            f"Only {len(df)} valid rows found. "
            "At least 5 are required."
        )

    if df["state"].nunique() < 2:
        raise ValueError(
            "Training data must contain at least 2 states."
        )

    print("\nTraining CSV:", CSV_FILE)
    print("Training rows:", len(df))
    print("\nState distribution:")
    print(df["state"].value_counts())

    return df


# ==========================================
# STEP 4: TRAIN THREE ML MODELS
# ==========================================

def train_models(df):
    X = df[FEATURES]
    y = df["state"]

    models = {
        "Decision_Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),
        "Naive_Bayes": GaussianNB()
    }

    for name, model in models.items():
        model.fit(X, y)

        model_path = BASE_DIR / f"{name.lower()}.pkl"

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        print(f"{name} trained and saved.")

    return models


# ==========================================
# STEP 5: READ DAY 1 SENSOR LOG
# ==========================================

def read_log_file():
    if not LOG_FILE.exists():
        raise FileNotFoundError(
            f"Log file not found: {LOG_FILE}"
        )

    records = []

    pattern = (
        r"Reading:\s*(\{.*?\})\s*,?\s*"
        r"Rule:\s*([^,]+),\s*"
        r"Action:\s*(\S+)"
    )

    with open(LOG_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            match = re.search(pattern, line)

            if not match:
                continue

            try:
                sensors = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                print(f"Invalid sensor data on line {line_number}")
                continue

            if not isinstance(sensors, dict):
                continue

            sensors = {
                str(key).strip().lower(): value
                for key, value in sensors.items()
            }

            record = {
                "log_line": line_number,
                "logged_rule": match.group(2).strip().upper(),
                "logged_action": match.group(3).strip().upper()
            }

            for feature in FEATURES:
                try:
                    value = sensors.get(feature)
                    record[feature] = (
                        float(value) if value is not None else None
                    )
                except (TypeError, ValueError):
                    record[feature] = None

            record["logged_state"] = action_to_state(
                record["logged_action"]
            )

            if any(
                record[feature] is None
                or pd.isna(record[feature])
                for feature in FEATURES
            ):
                record["Decision_Tree"] = "NOT_PREDICTED"
                record["Random_Forest"] = "NOT_PREDICTED"
                record["Naive_Bayes"] = "NOT_PREDICTED"
                record["ml_majority"] = "NOT_PREDICTED"
                record["comparison"] = "SENSOR_DROPOUT"

            records.append(record)

    return records


# ==========================================
# STEP 6: PREDICT AND COMPARE
# ==========================================

def compare_predictions():
    df = load_training_data()
    models = train_models(df)
    records = read_log_file()

    if not records:
        print(
            "\nNo valid log records found. "
            "Check the format of day1_log.txt."
        )
        return

    for record in records:
        if record.get("comparison") == "SENSOR_DROPOUT":
            continue

        sample = pd.DataFrame(
            [[record[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        predictions = []

        for name, model in models.items():
            prediction = model.predict(sample)[0]
            record[name] = prediction
            predictions.append(prediction)

        # Majority vote
        counts = pd.Series(predictions).value_counts()

        if counts.iloc[0] >= 2:
            majority = counts.index[0]
        else:
            majority = "DISAGREEMENT"

        record["ml_majority"] = majority

        if record["logged_state"] == "UNKNOWN":
            record["comparison"] = "UNKNOWN_LOG_STATE"
        elif majority == "DISAGREEMENT":
            record["comparison"] = "MODEL_DISAGREEMENT"
        elif majority == record["logged_state"]:
            record["comparison"] = "MATCH"
        else:
            record["comparison"] = "DIFFERENT"

    # Save output in Day 2
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== RESULTS ==========")
    print("Log file:", LOG_FILE)
    print("Output file:", OUTPUT_FILE)
    print("Total records:", len(result))

    print("\nComparison summary:")
    print(result["comparison"].value_counts().to_string())

    columns = [
        "log_line",
        "logged_rule",
        "logged_action",
        "logged_state",
        "Decision_Tree",
        "Random_Forest",
        "Naive_Bayes",
        "ml_majority",
        "comparison"
    ]

    print("\nDetailed predictions:")
    print(result[columns].to_string(index=False))


# ==========================================
# STEP 7: RUN
# ==========================================

if __name__ == "__main__":
    try:
        compare_predictions()

    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        print("\nERROR:", error)
