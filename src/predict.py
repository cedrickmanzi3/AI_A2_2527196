
"""Make predictions from one JSON farm record."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from pydantic import ValidationError

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
VERIFICATION_CODE = "AI-A2-7196"


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python predict.py '
            '\'{"rainfall_mm":1000,...}\''
        )
        sys.exit(2)

    try:
        farm_input = json.loads(sys.argv[1])

        # Reuse the API's input validation rules.
        from src.app import FarmInput

        validated_farm = FarmInput(**farm_input)
        farm = pd.DataFrame([validated_farm.model_dump()])

        regression_model = joblib.load(
            MODELS_DIR / "regression_model.joblib"
        )
        classification_model = joblib.load(
            MODELS_DIR / "classification_model.joblib"
        )
        clustering_model = joblib.load(
            MODELS_DIR / "clustering_model.joblib"
        )
        clustering_preprocessor = joblib.load(
            MODELS_DIR / "clustering_preprocessor.joblib"
        )

        predicted_yield = float(regression_model.predict(farm)[0])
        high_risk = int(classification_model.predict(farm)[0])
        risk_probability = float(
            classification_model.predict_proba(farm)[0][1]
        )

        cluster_data = farm[
            [
                "rainfall_mm",
                "avg_temp_c",
                "soil_moisture_pct",
                "fertilizer_kg",
                "field_size_ha",
                "cooperative_support",
                "season",
            ]
        ].copy()

        cluster_data["season"] = cluster_data["season"].map(
            {"A": 0, "B": 1}
        )

        transformed = clustering_preprocessor.transform(cluster_data)
        cluster = int(clustering_model.predict(transformed)[0])

        result = {
            "verification_code": VERIFICATION_CODE,
            "predicted_yield_tons": round(predicted_yield, 2),
            "high_risk_prediction": high_risk,
            "high_risk_probability": round(risk_probability, 4),
            "cluster": cluster,
            "interpretation": {
                "high_risk_prediction": (
                    "High risk" if high_risk == 1 else "Lower risk"
                ),
                "cluster_note": (
                    "Cluster labels identify groups; "
                    "they are not risk levels."
                ),
            },
        }

        print(json.dumps(result, indent=2))

    except (json.JSONDecodeError, ValidationError) as error:
        print(f"Invalid farm JSON: {error}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, FileNotFoundError) as error:
        print(f"Required model or field unavailable: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
