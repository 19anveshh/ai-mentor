"""Data leakage demonstration: train + evaluate on same data vs train/test split."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X, y = load_iris(return_X_y=True)

# --- LEAKAGE: train and evaluate on ALL data (no split) ---
leakage_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, random_state=42)),
])
leakage_model.fit(X, y)
leakage_pred = leakage_model.predict(X)
leakage_acc = accuracy_score(y, leakage_pred)
leakage_f1 = f1_score(y, leakage_pred, average="macro")

# --- PROPER: train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
proper_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, random_state=42)),
])
proper_model.fit(X_train, y_train)
proper_pred = proper_model.predict(X_test)
proper_acc = accuracy_score(y_test, proper_pred)
proper_f1 = f1_score(y_test, proper_pred, average="macro")

print(f"Leakage (train+test on same data):  accuracy={leakage_acc:.4f}, f1={leakage_f1:.4f}")
print(f"Proper (train/test split):          accuracy={proper_acc:.4f}, f1={proper_f1:.4f}")
print(f"Inflation from leakage:             +{leakage_acc - proper_acc:+.4f} accuracy")

# Why this is misleading:
# When you train and evaluate on the same data, the model memorizes the training
# examples instead of learning generalizable patterns. The 0.9733 accuracy is
# inflated — it measures memory, not prediction power. The 0.9333 from a proper
# train/test split is the honest number because the model is graded on data it
# has never seen, which reflects real-world performance.
