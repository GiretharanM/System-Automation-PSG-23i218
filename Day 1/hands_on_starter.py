
import random
import time

LOG_FILE = "day1_log.txt"


# STEP 1 and STEP 6: Generate sensor data
def generate_sensor_data():
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(2, 10), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "current": round(random.uniform(1, 20), 2)
    }


# STEP 2, STEP 7 and STEP 8: Define automation rules
RULES = [
    {
        "name": "CRITICAL_TEMPERATURE",
        "priority": 0,
        "condition": lambda r: r["temperature"] > 90,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 2,
        "condition": lambda r: r["pressure"] > 8,
        "action": "PRESSURE_WARNING"
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 3,
        "condition": lambda r: r["vibration"] > 4,
        "action": "VIBRATION_WARNING"
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 15,
        "action": "CURRENT_WARNING"
    }
]


# STEP 3: Evaluate rules
def evaluate_rules(reading, rules):
    for rule in sorted(rules, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# STEP 4: Perform action
def take_action(action):
    actions = {
        "STOP_MACHINE": "High temperature! Stop the machine.",
        "PRESSURE_WARNING": "High pressure! Check the machine.",
        "VIBRATION_WARNING": "High vibration! Inspect the machine.",
        "CURRENT_WARNING": "High current! Inspect the electrical system.",
        "CONTINUE_NORMAL": "Machine operating normally."
    }

    return actions.get(action, "Unknown action")


# STEP 5: Log every decision
def log_decision(reading, rule_name, action):
    with open(LOG_FILE, "a") as file:
        file.write(
            f"Reading: {reading}, "
            f"Rule: {rule_name}, "
            f"Action: {action}\n"
        )


# Main program
def main():
    for tick in range(1, 11):
        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
