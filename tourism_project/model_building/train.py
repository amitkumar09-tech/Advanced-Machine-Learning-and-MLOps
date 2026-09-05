"""
Model Training and Registration with Experimentation Tracking
----------------------------------------------------------------
Loads the train/test splits produced by the previous pipeline job, tunes an
XGBoost classifier with GridSearchCV, logs every tuned parameter and the
resulting metrics to MLflow (headless, file-backed -- no server or UI is
needed inside the GitHub Actions runner), evaluates the best model, and
saves it so the pipeline can commit it to tourism_project/deployment/.
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

# SQLite-backed MLflow store written to the job workspace -- no tracking
# server needs to be running for this to work inside a CI runner, and the
# SQLite backend avoids the deprecation warnings recent MLflow versions
# raise against the plain file-store backend.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tourism-wellness-package")


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

    # Hyperparameter grid tuned via GridSearchCV, informed by the manual
    # exploration done earlier in the Experimentation and Tracking section.
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run(run_name="grid_search_xgboost"):
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        preds = best_model.predict(Xtest)
        acc = accuracy_score(ytest, preds)
        f1 = f1_score(ytest, preds)

        # Log every tuned hyperparameter and the resulting test metrics so
        # each pipeline run leaves a permanent, comparable experiment record.
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)

        print("Best parameters found by GridSearchCV:")
        print(best_params)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(classification_report(ytest, preds))

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"Best model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
