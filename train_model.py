"""Train the GlycoAID diabetes risk prediction model.

The dataset is loaded directly from the mandatory GitHub URL. The script
cleans invalid zero values, performs EDA, trains Logistic Regression and
Decision Tree models, compares them, and saves the best model with Joblib.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
TARGET = "Outcome"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
MODEL_PATH = BASE_DIR / "model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_dataset() -> pd.DataFrame:
    """Load the diabetes dataset directly from the required GitHub URL."""
    data = pd.read_csv(DATA_URL)
    missing_columns = set(FEATURES + [TARGET]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return data


def clean_invalid_values(data: pd.DataFrame) -> pd.DataFrame:
    """Replace medically invalid zero values with NaN for median imputation."""
    cleaned = data.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    return cleaned


def save_eda(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> None:
    """Save EDA summary and charts for the assignment evidence."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    target_counts = raw_data[TARGET].value_counts().sort_index()
    invalid_zero_counts = (raw_data[ZERO_AS_MISSING] == 0).sum().sort_values(ascending=False)

    summary = [
        "GlycoAID EDA Summary",
        "====================",
        f"Dataset source: {DATA_URL}",
        f"Rows: {raw_data.shape[0]}",
        f"Columns: {raw_data.shape[1]}",
        "",
        "Target distribution:",
        target_counts.to_string(),
        "",
        "Invalid zero counts handled with median imputation:",
        invalid_zero_counts.to_string(),
        "",
        "Descriptive statistics after cleaning invalid zeros:",
        cleaned_data[FEATURES].describe().round(3).to_string(),
    ]
    (OUTPUT_DIR / "eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=raw_data,
        x=TARGET,
        hue=TARGET,
        palette=["#2e7d6b", "#c7504a"],
        legend=False,
    )
    plt.title("Diabetes Outcome Distribution")
    plt.xlabel("Outcome (0 = Low Risk, 1 = High Risk)")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "outcome_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 7))
    correlation = cleaned_data[FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="BrBG", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    cleaned_data[FEATURES].hist(figsize=(11, 8), bins=24, color="#4d8f8b", edgecolor="white")
    plt.suptitle("Feature Distributions After Invalid Zero Cleaning")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_distributions.png", dpi=160)
    plt.close()


def build_pipeline(model_name: str) -> Pipeline:
    """Create a preprocessing and classification pipeline."""
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "Decision Tree":
        classifier = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    numeric_steps = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_steps, FEATURES)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a binary classifier."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def get_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature importance or absolute coefficient values."""
    estimator = best_pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(FEATURES))

    importance = pd.DataFrame({"feature": FEATURES, "importance": values})
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance(importance: pd.DataFrame) -> None:
    """Save feature importance table and chart."""
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, x="importance", y="feature", color="#2e7d6b")
    plt.title("Best Model Feature Importance")
    plt.xlabel("Relative Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset()
    cleaned_data = clean_invalid_values(raw_data)
    save_eda(raw_data, cleaned_data)

    x = cleaned_data[FEATURES]
    y = cleaned_data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    trained_models: dict[str, Pipeline] = {}
    results = []

    for model_name in ["Logistic Regression", "Decision Tree"]:
        pipeline = build_pipeline(model_name)
        pipeline.fit(x_train, y_train)
        trained_models[model_name] = pipeline
        results.append(evaluate_model(model_name, pipeline, x_test, y_test))

    comparison = pd.DataFrame(results).sort_values(
        by=["roc_auc", "f1_score", "recall"],
        ascending=False,
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = trained_models[best_name]
    joblib.dump(best_pipeline, MODEL_PATH)

    predictions = best_pipeline.predict(x_test)
    report = classification_report(y_test, predictions, target_names=["Low Risk", "High Risk"])
    matrix = confusion_matrix(y_test, predictions)

    importance = get_feature_importance(best_pipeline)
    save_feature_importance(importance)

    metadata = {
        "project": "GlycoAID - Diabetes Risk Prediction System",
        "dataset_url": DATA_URL,
        "features": FEATURES,
        "zero_as_missing": ZERO_AS_MISSING,
        "selected_model": best_name,
        "model_path": str(MODEL_PATH),
        "model_comparison": comparison.round(4).to_dict(orient="records"),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "top_features": importance.head(5).round(4).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Selected model: {best_name}")
    print(comparison.round(4).to_string(index=False))
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
