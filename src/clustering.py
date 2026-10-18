
"""Compare KMeans cluster counts without fitting preprocessing on test data."""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_validation import clean_data

SEED = 7196

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
    "season",
]


def make_clustering_preprocessor():
    """Create the imputer and scaler pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def prepare_features(df):
    """Encode season without using prediction targets."""
    X = df[FEATURES].copy()
    X["season"] = X["season"].map({"A": 0, "B": 1})
    return X


def train_clustering(df, project_dir):
    """Split first, fit preprocessing on training data, and compare K."""
    df = clean_data(df).reset_index(drop=True)
    X = prepare_features(df)

    X_train, X_test = train_test_split(
        X,
        test_size=0.20,
        random_state=SEED,
    )

    preprocessor = make_clustering_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    scores = {}
    models = {}

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
        )
        train_labels = model.fit_predict(X_train_scaled)

        scores[str(k)] = float(
            silhouette_score(X_train_scaled, train_labels)
        )
        models[k] = (model, train_labels)

    selected_k = int(max(scores, key=scores.get))
    selected_model, train_labels = models[selected_k]

    # Apply the selected model to held-out test rows.
    test_labels = selected_model.predict(X_test_scaled)

    # Profile the training clusters using their original feature values.
    profile_df = X_train.copy()
    profile_df["cluster"] = train_labels

    profiles = {}
    for cluster_id, group in profile_df.groupby("cluster"):
        profiles[str(int(cluster_id))] = {
            "farm_count": int(len(group)),
            "mean_rainfall_mm": float(group["rainfall_mm"].mean()),
            "mean_temperature_c": float(group["avg_temp_c"].mean()),
            "mean_soil_moisture_pct": float(
                group["soil_moisture_pct"].mean()
            ),
            "mean_fertilizer_kg": float(group["fertilizer_kg"].mean()),
            "mean_field_size_ha": float(group["field_size_ha"].mean()),
            "season_A_fraction": float((group["season"] == 0).mean()),
            "cooperative_support_fraction": float(
                group["cooperative_support"].mean()
            ),
        }

    output_dir = Path(project_dir) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "verification_code": "AI-A2-7196",
        "seed": SEED,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "tested_k": [2, 3, 4, 5, 6],
        "silhouette_scores": scores,
        "selected_k": selected_k,
        "selection_reason": (
            "Highest silhouette score on the training set."
        ),
        "test_cluster_counts": {
            str(int(k)): int((test_labels == k).sum())
            for k in sorted(set(test_labels))
        },
        "cluster_profiles": profiles,
    }

    (output_dir / "clustering_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report, selected_model, preprocessor, train_labels


def main():
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "AI_A2_2527196.csv"

    df = pd.read_csv(csv_path)
    report, _, _, _ = train_clustering(df, project_dir)

    print("CLUSTERING MODEL COMPARISON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
