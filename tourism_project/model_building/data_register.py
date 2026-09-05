"""
Data Registration
------------------
Reads the raw tourism dataset from the repository's data folder, validates
that every column the downstream pipeline depends on is present, and prints
a short summary so the dataset is "registered" (checked and understood)
before it moves further down the pipeline. This script is the first job
run by the GitHub Actions workflow.
"""

import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Columns the rest of the pipeline (prep.py, train.py) expects to find.
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and validate its schema before it is used further."""
    df = pd.read_csv(path)

    # Fail fast (and loudly, in the Actions log) if the schema ever drifts.
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    print("Dataset registered successfully.")
    print(f"Source path : {path}")
    print(f"Shape       : {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nTarget class balance (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(3).rename("proportion"))

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    print("\nColumns with missing values:")
    print(missing_counts if not missing_counts.empty else "None")

    return df


if __name__ == "__main__":
    register_dataset()
