
"""Train final models and save them for local inference."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_validation import clean_data
from src.preprocessing import make_preprocessor
from src.clustering import (
    FEATURES,
    make_clustering_preprocessor,
    prepare_features,
)

SEED = 7196


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "data" / "AI_A2_2527196.csv"
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(pd.read_csv(data_path)).reset_index(drop=True)

    # Regression: selected model from the measured comparison.
    X_reg = df.drop(columns=["yield_tons", "high_risk"])
    y_reg = df["yield_tons"]

    regression_model = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LinearRegression()),
    ])
    regression_model.fit(X_reg, y_reg)

    joblib.dump(
        regression_model,
        models_dir / "regression_model.joblib",
    )

    # Classification: selected model from the measured comparison.
    X_cls = df.drop(columns=["yield_tons", "high_risk"])
    y_cls = df["high_risk"]

    classification_model = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    classification_model.fit(X_cls, y_cls)

    joblib.dump(
        classification_model,
        models_dir / "classification_model.joblib",
    )

    # Clustering: fit preprocessing and KMeans on the cleaned input features.
    X_cluster = prepare_features(df)
    cluster_preprocessor = make_clustering_preprocessor()
    X_scaled = cluster_preprocessor.fit_transform(X_cluster)

    from sklearn.cluster import KMeans

    clustering_model = KMeans(
        n_clusters=2,
        random_state=SEED,
        n_init=10,
    )
    clustering_model.fit(X_scaled)

    joblib.dump(
        cluster_preprocessor,
        models_dir / "clustering_preprocessor.joblib",
    )
    joblib.dump(
        clustering_model,
        models_dir / "clustering_model.joblib",
    )

    print("MODELS SAVED SUCCESSFULLY")
    for path in sorted(models_dir.glob("*.joblib")):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
