import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")


def load_data(data_path: str) -> pd.DataFrame:
    """Load preprocessed Telco Churn dataset."""
    if os.path.isdir(data_path):
        files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_path = os.path.join(data_path, files[0])

    df = pd.read_csv(data_path)
    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame):
    """Split features and target, then train/test split."""
    target_col = "Churn"

    # Drop customerID if exists
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target if string
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_and_log(
    X_train, X_test, y_train, y_test,
    n_estimators: int = 100,
    max_depth: int = None,
    model_name: str = "telco-churn-model"
):
    """Train Random Forest and log to MLflow."""

    with mlflow.start_run(run_name="RandomForest_TelcoChurn"):
        # ── Parameters ──────────────────────────────────────────────
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        # ── Train ────────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # ── Predict ──────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ── Metrics ──────────────────────────────────────────────────
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)

        print(f"\n[METRICS]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")

        # ── Classification Report as artifact ────────────────────────
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # ── Confusion Matrix as artifact ─────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        cm_path = "confusion_matrix.txt"
        with open(cm_path, "w") as f:
            f.write("Confusion Matrix\n")
            f.write(f"TN={cm[0,0]}  FP={cm[0,1]}\n")
            f.write(f"FN={cm[1,0]}  TP={cm[1,1]}\n")
        mlflow.log_artifact(cm_path)

        # ── Feature Importance as artifact ───────────────────────────
        feat_imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        feat_path = "feature_importance.csv"
        feat_imp.to_csv(feat_path, index=False)
        mlflow.log_artifact(feat_path)

        # ── Log Model ────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X_train.iloc[:5]
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n[INFO] Run ID: {run_id}")
        print(f"[INFO] Model logged as '{model_name}'")
        return model, run_id


def main():
    parser = argparse.ArgumentParser(description="Train Telco Churn Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./MLProject/telcocustomerchurn_preprocessing.csv",
        help="Path to preprocessed CSV file or folder"
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="telco-churn-model")
    args = parser.parse_args()

    print("=" * 50)
    print("  Telco Churn — MLflow Training Pipeline")
    print("=" * 50)

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = preprocess(df)
    train_and_log(
        X_train, X_test, y_train, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()