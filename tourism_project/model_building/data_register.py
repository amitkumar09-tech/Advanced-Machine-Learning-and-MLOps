"""
Data Registration
------------------
Reads the raw tourism dataset from the repository's data folder, validates
that every expected column is present, and prints a short summary so the
dataset can be treated as "registered" before it moves further down the
pipeline.
"""

import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    print("Dataset registered successfully.")
    print(f"Source: {path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nTarget class balance (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(3).rename("proportion"))

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    print("\nColumns with missing values:")
    print(missing_counts if not missing_counts.empty else "None")

    return df


if __name__ == "__main__":
    register_dataset()
