"""Model wrapper for sklearn classifiers.

Production pattern: Scale -> Train -> Predict -> Evaluate.
Reusable across RAG, fine-tuning, and any ML project.
"""

from typing import Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


class ModelWrapper:
    """Wrapper around sklearn pipeline with train/predict/evaluate.

    Attributes:
        pipeline: Sklearn Pipeline containing scaler + classifier.
    """

    def __init__(self) -> None:
        """Initialize the pipeline with StandardScaler + LogisticRegression.

        StandardScaler normalizes features so none dominates.
        LogisticRegression learns the pattern from data.
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ])

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on training data.

        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (n_samples,).
        """
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data.

        Args:
            X: Input features (n_samples, n_features).

        Returns:
            Predicted class labels.
        """
        return self.pipeline.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model and return metrics.

        Args:
            X: Test features (n_samples, n_features).
            y: True labels (n_samples,).

        Returns:
            Dictionary with 'accuracy' and 'f1_macro' scores.
        """
        y_pred = self.pipeline.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro")),
        }


if __name__ == "__main__":
    # Quick smoke test with iris dataset
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    wrapper = ModelWrapper()
    wrapper.train(X_train, y_train)
    metrics = wrapper.evaluate(X_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")