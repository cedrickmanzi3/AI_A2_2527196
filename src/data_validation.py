
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
]


def validate_data(df: pd.DataFrame) -> None:
    """Check required columns, ranges, and target values."""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not df["rainfall_mm"].between(600, 1400).all():
        raise ValueError("Rainfall must be between 600 and 1400 mm.")

    if not df["avg_temp_c"].between(12, 26).all():
        raise ValueError("Temperature must be between 12 and 26 C.")

    if not df["field_size_ha"].between(0.2, 4.0).all():
        raise ValueError("Field size must be between 0.2 and 4.0 ha.")

    if not df["season"].isin(["A", "B"]).all():
        raise ValueError("Season must be A or B.")

    if not df["cooperative_support"].isin([0, 1]).all():
        raise ValueError("Cooperative support must be 0 or 1.")

    if not df["high_risk"].isin([0, 1]).all():
        raise ValueError("High risk must be 0 or 1.")

    if (df["yield_tons"].dropna() < 0.5).any():
        raise ValueError("Yield must be at least 0.5 tons.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates while preserving missing values for pipelines."""
    validate_data(df)

    cleaned = df.drop_duplicates().reset_index(drop=True)

    return cleaned


def main():
    """Load, validate, and summarize the generated dataset."""
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "AI_A2_2527196.csv"

    df = pd.read_csv(csv_path)

    print("DATA VALIDATION")
    print("-" * 40)
    print(f"Original rows: {len(df)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    cleaned = clean_data(df)

    print(f"Rows after duplicate removal: {len(cleaned)}")
    print("\nMissing values after duplicate removal:")
    print(cleaned.isna().sum())
    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()
