
"""Run the Musanze Harvest Intelligence ML workflow."""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Validate data", "src.data_validation"),
    ("Preprocess data", "src.preprocessing"),
    ("Compare regression models", "src.regression"),
    ("Compare classification models", "src.classification"),
    ("Compare clustering models", "src.clustering"),
    ("Save final models", "src.save_models"),
]


def main():
    print("=" * 55)
    print("MUSANZE HARVEST INTELLIGENCE")
    print("Verification code: AI-A2-7196")
    print("=" * 55)

    for description, module in STEPS:
        print(f"\n\n>>> {description}")

        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_DIR,
            check=False,
        )

        if result.returncode != 0:
            print(f"\nERROR: {description} failed.")
            sys.exit(result.returncode)

    print("\nWorkflow completed successfully.")
    print("Dataset, evaluation reports, and trained models are available.")


if __name__ == "__main__":
    main()
