
"""Train and compare high-risk classification models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_validation import clean_data
from src.preprocessing import make_preprocessor

SEED = 7196


def classification_metrics(y_true, predictions, probabilities=None):
    results = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }

    if probabilities is not None and len(np.unique(y_true)) == 2:
        results["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )

    return results


def train_classification(df, project_dir):
    df = clean_data(df)

    # Do not use yield_tons or high_risk as input features.
    features = df.drop(columns=["yield_tons", "high_risk"])
    target = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=SEED,
        stratify=target,
    )

    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]

        results[name] = classification_metrics(
            y_test, predictions, probabilities
        )

        results[name]["classification_report"] = (
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
        )

        fitted_models[name] = pipeline

    selected_name = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: results[name]["f1"],
    )

    output_dir = Path(project_dir) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "verification_code": "AI-A2-7196",
        "seed": SEED,
        "test_rows": int(len(y_test)),
        "positive_class": 1,
        "positive_class_meaning": "High risk",
        "metrics": results,
        "selected_model": selected_name,
        "selection_reason": (
            "Highest F1-score among the two candidate classifiers "
            "on the held-out test set."
        ),
    }

    output_path = output_dir / "classification_metrics.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report, fitted_models, (X_test, y_test)


def main():
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "AI_A2_2527196.csv"

    df = pd.read_csv(csv_path)
    report, _, _ = train_classification(df, project_dir)

    print("CLASSIFICATION MODEL COMPARISON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

