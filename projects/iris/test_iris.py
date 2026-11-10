"""Test script demonstrating ModelWrapper usage on iris dataset."""

from model_wrapper import ModelWrapper
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main() -> None:
    # 1. Load data
    X, y = load_iris(return_X_y=True)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 3. Train
    wrapper = ModelWrapper()
    wrapper.train(X_train, y_train)
    print("Training complete")

    # 4. Predict on single sample
    sample = X_test[0:1]
    prediction = wrapper.predict(sample)
    print(f"Sample prediction: class {prediction[0]} (actual: {y_test[0]})")

    # 5. Evaluate
    metrics = wrapper.evaluate(X_test, y_test)
    print(f"\nReport Card:")
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"  F1 Macro : {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()