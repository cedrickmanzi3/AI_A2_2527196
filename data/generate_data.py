
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


# Student-specific assignment details
SEED = 7196
VERIFICATION_CODE = "AI-A2-7196"

# Find the main project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

CSV_PATH = DATA_DIR / "AI_A2_2527196.csv"
FINGERPRINT_PATH = ARTIFACTS_DIR / "data_fingerprint.txt"


def generate_data():
    """Generate the personalized synthetic potato-farm dataset."""
    rng = np.random.default_rng(SEED)

    # Generate the required number of base rows.
    base_rows = 240 + (SEED % 41)

    df = pd.DataFrame({
        "rainfall_mm": rng.uniform(600, 1400, base_rows),
        "avg_temp_c": rng.uniform(12, 26, base_rows),
        "soil_moisture_pct": rng.uniform(20, 80, base_rows),
        "fertilizer_kg": rng.uniform(0, 250, base_rows),
        "field_size_ha": rng.uniform(0.2, 4.0, base_rows),
        "season": rng.choice(["A", "B"], base_rows),
        "cooperative_support": rng.integers(0, 2, base_rows),
    })

    # Apply the required harvest formula.
    season_effect = np.where(df["season"] == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, base_rows)

    df["yield_tons"] = (
        0.004 * df["rainfall_mm"]
        + 0.06 * df["soil_moisture_pct"]
        + 0.02 * df["fertilizer_kg"]
        + 1.30 * df["field_size_ha"]
        - 0.25 * abs(df["avg_temp_c"] - 18)
        + 0.70 * df["cooperative_support"]
        + season_effect
        + noise
    ).clip(lower=0.50)

    # High risk means yield per hectare is below the median.
    yield_per_hectare = df["yield_tons"] / df["field_size_ha"]
    df["high_risk"] = (
        yield_per_hectare < yield_per_hectare.median()
    ).astype(int)

    # Introduce the missing values required by the assignment.
    missing_count = round(0.05 * base_rows)

    soil_positions = rng.choice(
        base_rows, size=missing_count, replace=False
    )
    fertilizer_positions = rng.choice(
        base_rows, size=missing_count, replace=False
    )

    df.loc[soil_positions, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_positions, "fertilizer_kg"] = np.nan

    # Append three exact duplicate rows.
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)

    return df


def main():
    """Save the dataset and its SHA-256 fingerprint."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_data()
    df.to_csv(CSV_PATH, index=False, float_format="%.4f")

    # Hash the exact bytes written to the CSV.
    fingerprint = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

    FINGERPRINT_PATH.write_text(
        f"Student verification code: {VERIFICATION_CODE}\n"
        f"Random seed: {SEED}\n"
        f"CSV file: {CSV_PATH.name}\n"
        f"SHA-256: {fingerprint}\n",
        encoding="utf-8",
    )

    print("=" * 55)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print("=" * 55)
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Random seed: {SEED}")
    print(f"Base records: {240 + (SEED % 41)}")
    print(f"Final records: {len(df)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing soil moisture: {df['soil_moisture_pct'].isna().sum()}")
    print(f"Missing fertilizer: {df['fertilizer_kg'].isna().sum()}")
    print(f"Dataset saved: {CSV_PATH}")
    print(f"SHA-256: {fingerprint}")
    print("=" * 55)


if __name__ == "__main__":
    main()
