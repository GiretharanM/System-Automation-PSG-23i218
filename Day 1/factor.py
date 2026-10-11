import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# STEP 1: Load dataset
df = pd.read_csv("day2_sample.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print(df.head())

# STEP 2: Separate input features and target
X = df[[
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm"
]]

y = df["state"]

# STEP 3: Convert machine states into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# STEP 4: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 5: Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

# STEP 6: Train the model
model.fit(X_train, y_train)

print("\nDecision Tree training completed!")

# STEP 7: Test the trained model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))

# STEP 8: Save the trained model
with open("trained_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("\nModel saved as trained_model.pkl")
print("Label encoder saved as label_encoder.pkl")

# STEP 9: Predict a new machine condition
new_data = pd.DataFrame([{
    "temperature": 85.0,
    "pressure": 5.0,
    "vibration": 2.5,
    "current": 12.0,
    "rpm": 1800
}])

prediction = model.predict(new_data)
predicted_state = encoder.inverse_transform(prediction)

print("\nNew Sensor Readings:")
print(new_data)

print("Predicted Machine State:", predicted_state[0])
