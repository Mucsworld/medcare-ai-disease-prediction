"""Train the GlycoAID diabetes risk prediction model.

This script fetches the mandatory dataset directly from GitHub, performs
basic EDA, trains Logistic Regression and Decision Tree models, compares
their performance, and saves the best model as model.pkl.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_dataset() -> pd.DataFrame:
    """Load the required diabetes dataset directly from GitHub."""
    df = pd.read_csv(DATA_URL)
    missing_columns = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df


def clean_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace medically invalid zero values with NaN for later imputation."""
    cleaned = df.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    return cleaned


def save_eda(df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Create a compact EDA report and visualizations."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    outcome_counts = df[TARGET].value_counts().sort_index()
    zero_counts = (df[ZERO_AS_MISSING] == 0).sum().sort_values(ascending=False)

    summary = [
        "GlycoAID EDA Summary",
        "====================",
        f"Dataset source: {DATA_URL}",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "Target distribution:",
        outcome_counts.to_string(),
        "",
        "Invalid zero counts replaced with median imputation:",
        zero_counts.to_string(),
        "",
        "Cleaned descriptive statistics:",
        cleaned_df[FEATURES].describe().round(3).to_string(),
    ]
    (OUTPUT_DIR / "eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette=["#2e7d6b", "#c7504a"], legend=False)
    plt.title("Diabetes Outcome Distribution")
    plt.xlabel("Outcome (0 = Low Risk, 1 = High Risk)")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "outcome_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 7))
    corr = cleaned_df[FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    cleaned_df[FEATURES].hist(figsize=(11, 8), bins=24, color="#4d8f8b", edgecolor="white")
    plt.suptitle("Feature Distributions After Invalid Zero Cleaning")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_distributions.png", dpi=160)
    plt.close()


def make_model_pipeline(model_name: str) -> Pipeline:
    """Build preprocessing plus classifier pipeline."""
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

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessor, FEATURES)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model with common binary classification metrics."""
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


def extract_feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    """Return feature importance or coefficients for the selected model."""
    estimator = best_pipeline.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        label = "importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
        label = "importance"
    else:
        values = np.zeros(len(FEATURES))
        label = "importance"

    importance = pd.DataFrame({"feature": FEATURES, label: values})
    importance = importance.sort_values(label, ascending=False).reset_index(drop=True)
    return importance


def save_feature_importance(importance: pd.DataFrame) -> None:
    """Save feature importance as CSV and chart."""
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

    df = load_dataset()
    cleaned_df = clean_invalid_zeros(df)
    save_eda(df, cleaned_df)

    x = cleaned_df[FEATURES]
    y = cleaned_df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_names = ["Logistic Regression", "Decision Tree"]
    trained_models: dict[str, Pipeline] = {}
    metrics = []

    for name in model_names:
        pipeline = make_model_pipeline(name)
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        metrics.append(evaluate_model(name, pipeline, x_test, y_test))

    comparison = pd.DataFrame(metrics).sort_values(
        by=["roc_auc", "f1_score", "recall"],
        ascending=False,
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.iloc[0]["model"]
    best_pipeline = trained_models[best_name]
    joblib.dump(best_pipeline, "model.pkl")

    predictions = best_pipeline.predict(x_test)
    report = classification_report(y_test, predictions, target_names=["Low Risk", "High Risk"])
    matrix = confusion_matrix(y_test, predictions)

    importance = extract_feature_importance(best_pipeline)
    save_feature_importance(importance)

    metadata = {
        "project": "GlycoAID - Diabetes Risk Prediction System",
        "dataset_url": DATA_URL,
        "features": FEATURES,
        "zero_as_missing": ZERO_AS_MISSING,
        "selected_model": str(best_name),
        "model_comparison": comparison.round(4).to_dict(orient="records"),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "top_features": importance.head(5).round(4).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Selected model: {best_name}")
    print(comparison.round(4).to_string(index=False))
    print("\nClassification report for selected model:")
    print(report)
    print("Saved: model.pkl")


if __name__ == "__main__":
    main()
