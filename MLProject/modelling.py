import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import warnings
warnings.filterwarnings("ignore")


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset."""

    print(f"[DEBUG] Current directory : {os.getcwd()}")
    print(f"[DEBUG] Data path         : {data_path}")
    print(f"[DEBUG] Absolute path     : {os.path.abspath(data_path)}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}"
        )

    df = pd.read_csv(data_path)

    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")

    return df


def preprocess(df: pd.DataFrame):
    """Prepare train-test split."""

    target_col = "Churn"

    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"[INFO] Train size: {X_train.shape}")
    print(f"[INFO] Test size : {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_and_log(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=100,
    max_depth=None,
    model_name="telco-churn-model",
):

    with mlflow.start_run(run_name="RandomForest_TelcoChurn"):

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
        )

        with open("classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

        cm = confusion_matrix(y_test, y_pred)

        with open("confusion_matrix.txt", "w") as f:
            f.write("Confusion Matrix\n")
            f.write(f"TN={cm[0,0]} FP={cm[0,1]}\n")
            f.write(f"FN={cm[1,0]} TP={cm[1,1]}\n")

        mlflow.log_artifact("confusion_matrix.txt")

        feat_imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        feat_imp.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X_train.iloc[:5],
        )

        run_id = mlflow.active_run().info.run_id

        print(f"[INFO] Run ID: {run_id}")

        return model, run_id


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="telcocustomerchurn_preprocessing.csv",
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="telco-churn-model",
    )

    args = parser.parse_args()

    if args.max_depth == 0:
        args.max_depth = None

    print("=" * 50)
    print("Telco Churn — MLflow Training Pipeline")
    print("=" * 50)

    df = load_data(args.data_path)

    X_train, X_test, y_train, y_test = preprocess(df)

    train_and_log(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()