
import pandas as pd
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv("m_s_d.csv")

print("Dataset loaded:", df.shape)



features = [
    "temperature",
    "pressure",
    "vibration",
    "current"
]

X = df[features].copy()
X = X.fillna(X.median())

y = df["action"].copy()
y = y.fillna(y.mode()[0])



nb = GaussianNB()
nb.fit(X, y)

print("\nNaive Bayes training completed!")



dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

print("Decision Tree training completed!")



rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X, y)

print("Random Forest training completed!")



models = {
    "naive_bayes.pkl": nb,
    "decision_tree.pkl": dt,
    "random_forest.pkl": rf
}

for filename, model in models.items():
    with open(filename, "wb") as f:
        pickle.dump(model, f)

print("\nAll models saved successfully!")


new_data = pd.DataFrame([{
    "temperature": 85.0,
    "pressure": 5.0,
    "vibration": 2.5,
    "current": 12.0
}])



print("\n===== PREDICTIONS =====")

nb_prediction = nb.predict(new_data)[0]
dt_prediction = dt.predict(new_data)[0]
rf_prediction = rf.predict(new_data)[0]

print("Naive Bayes:", nb_prediction)
print("Decision Tree:", dt_prediction)
print("Random Forest:", rf_prediction)


result = new_data.copy()

result["Naive_Bayes"] = nb_prediction
result["Decision_Tree"] = dt_prediction
result["Random_Forest"] = rf_prediction

result.to_csv("prediction_results.csv", index=False)

print("\nPrediction results saved successfully!")
