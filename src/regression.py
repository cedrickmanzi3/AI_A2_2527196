
"""Train and compare harvest-yield regression models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor
from src.data_validation import clean_data


SEED = 7196


def regression_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_regression(df, project_dir):
    """Compare a mean baseline and two regression models."""
    df = clean_data(df)

    features = df.drop(columns=["yield_tons", "high_risk"])
    target = df["yield_tons"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=SEED,
    )

    baseline_predictions = np.full(len(y_test), y_train.mean())
    results = {
        "mean_baseline": regression_metrics(y_test, baseline_predictions)
    }

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    fitted_models = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results[name] = regression_metrics(y_test, predictions)
        fitted_models[name] = pipeline

    selected_name = min(
        candidates,
        key=lambda name: results[name]["mae"],
    )

    output_dir = Path(project_dir) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "verification_code": "AI-A2-7196",
        "seed": SEED,
        "test_rows": len(y_test),
        "metrics": results,
        "selected_model": selected_name,
        "selection_reason": (
            "Lowest MAE among the two candidate models on the held-out test set."
        ),
    }

    (output_dir / "regression_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report, fitted_models, (X_test, y_test)


def main():
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "AI_A2_2527196.csv"

    df = pd.read_csv(csv_path)
    report, _, _ = train_regression(df, project_dir)

    print("REGRESSION MODEL COMPARISON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
