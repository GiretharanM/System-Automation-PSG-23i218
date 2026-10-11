
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("day2_sample.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print(df.head())

features = [
    "temperature", "pressure", "vibration",
    "current", "rpm"
]

X = df[features]
y = df["state"]

X = X.fillna(X.median())
y = y.fillna(y.mode()[0])

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n===== NAIVE BAYES =====")
print("Accuracy:",
      round(accuracy_score(y_test, nb_pred) * 100, 2), "%")

print(classification_report(
    y_test,
    nb_pred,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))


dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\n===== DECISION TREE =====")
print("Accuracy:",
      round(accuracy_score(y_test, dt_pred) * 100, 2), "%")

print(classification_report(
    y_test,
    dt_pred,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:",
      round(accuracy_score(y_test, rf_pred) * 100, 2), "%")

print(classification_report(
    y_test,
    rf_pred,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(
    n_clusters=len(encoder.classes_),
    random_state=42,
    n_init=10
)

kmeans.fit(X_train_scaled)

clusters = kmeans.predict(X_test_scaled)

print("\n===== K-MEANS =====")
print("Number of clusters:", kmeans.n_clusters)

cluster_results = pd.DataFrame({
    "Actual_State": encoder.inverse_transform(y_test),
    "Cluster": clusters
})

print("\nCluster assignments:")
print(cluster_results.head(20))

print("\nActual states in each cluster:")
print(pd.crosstab(
    cluster_results["Cluster"],
    cluster_results["Actual_State"]
))


results = pd.DataFrame({
    "Algorithm": [
        "Naive Bayes",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy (%)": [
        accuracy_score(y_test, nb_pred) * 100,
        accuracy_score(y_test, dt_pred) * 100,
        accuracy_score(y_test, rf_pred) * 100
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results.round(2).to_string(index=False))


with open("naive_bayes.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("decision_tree.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nAll models saved successfully!")


new_data = pd.DataFrame([{
    "temperature": 85.0,
    "pressure": 5.0,
    "vibration": 2.5,
    "current": 12.0,
    "rpm": 1800
}])

print("\n===== NEW SENSOR PREDICTION =====")
print(new_data)

for name, model in [
    ("Naive Bayes", nb_model),
    ("Decision Tree", dt_model),
    ("Random Forest", rf_model)
]:
    prediction = model.predict(new_data)
    state = encoder.inverse_transform(prediction)[0]
    print(name, "Prediction:", state)

new_scaled = scaler.transform(new_data)
cluster = kmeans.predict(new_scaled)[0]

print("K-Means Cluster:", cluster)
