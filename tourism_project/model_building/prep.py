"""
Data Preparation
-----------------
Loads the registered dataset from the repository's data folder, cleans it,
drops columns that carry no predictive signal, and splits it into stratified
train/test sets. The splits are written as CSV files in the current working
directory so the GitHub Actions workflow can pass them to the next job as a
workflow artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

# Columns that are identifiers / row indices and have no predictive value
UNNECESSARY_COLUMNS = ["Unnamed: 0", "CustomerID"]

TARGET_COL = "ProdTaken"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier / index columns
    drop_cols = [c for c in UNNECESSARY_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Fix a known data-entry inconsistency in Gender ("Fe Male" -> "Female")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Drop exact duplicate rows, if any
    df = df.drop_duplicates()

    return df


def prepare_data(path: str = RAW_PATH):
    df = pd.read_csv(path)
    df = clean_data(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Train shape: {Xtrain.shape}  |  Test shape: {Xtest.shape}")
    print(f"Train target balance:\n{ytrain.value_counts(normalize=True).round(3)}")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
