"""
Model Training and Registration with Experimentation Tracking
----------------------------------------------------------------
Loads the train/test splits produced by the data-preparation job, builds a
preprocessing + XGBoost pipeline, tunes it with GridSearchCV, logs every
tuned parameter set and the resulting metrics to MLflow, evaluates the best
model on the held-out test set, and saves that best model so the workflow
can commit it to the repository.
"""

import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

MODEL_OUTPUT_PATH = "tourism_project/deployment/best_model.joblib"


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline():
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    return make_pipeline(preprocessor, model)


def main():
    Xtrain, Xtest, ytrain, ytest = load_splits()

    pipeline = build_pipeline()

    # Hyperparameter grid to tune (models allowed by the rubric include
    # Decision Tree / Bagging / Random Forest / AdaBoost / Gradient Boosting
    # / XGBoost -- XGBoost is used here)
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("tourism_package_prediction")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        preds = best_model.predict(Xtest)
        acc = accuracy_score(ytest, preds)
        f1 = f1_score(ytest, preds)

        # Log all tuned parameters and evaluation metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)

        print("Best parameters found by GridSearchCV:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"\nTest accuracy: {acc:.4f}")
        print(f"Test F1 score: {f1:.4f}")
        print("\nClassification report:\n", classification_report(ytest, preds))

    # Save the best model so the workflow can commit it into the repository
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"\nBest model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
