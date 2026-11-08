"""Week 1 baseline: sklearn classification pipeline with eval report."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
from pathlib import Path

# 1. Load data
X, y = load_iris(return_X_y=True, as_frame=True)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocess + train
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_s, y_train)

# 4. Evaluate
y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

report = {
    "model": "LogisticRegression",
    "dataset": "iris",
    "test_size": 0.2,
    "accuracy": round(acc, 4),
    "f1_macro": round(f1, 4),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
}

Path("EVAL_REPORTS").mkdir(exist_ok=True)
with open("EVAL_REPORTS/week1_baseline.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
print("Eval report saved to EVAL_REPORTS/week1_baseline.json")
