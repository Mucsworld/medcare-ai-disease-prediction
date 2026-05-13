import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
DATASET_PATH = Path("disease_dataset.csv")
MODEL_PATH = Path("disease_prediction_model.joblib")
ENCODERS_PATH = Path("label_encoders.joblib")
BUNDLE_PATH = Path("model_bundle.joblib")
METRICS_PATH = Path("model_metrics.txt")

FEATURE_COLUMNS = ["Age", "Gender", "Fever", "Cough", "Headache", "Body_Pain"]
TARGET_COLUMN = "Disease"
DISEASE_CLASSES = ["Malaria", "Flu", "Typhoid", "Cold"]

SYMPTOM_PROFILES = {
    "Malaria": {"Fever": 0.92, "Cough": 0.20, "Headache": 0.78, "Body_Pain": 0.86},
    "Flu": {"Fever": 0.74, "Cough": 0.82, "Headache": 0.56, "Body_Pain": 0.68},
    "Typhoid": {"Fever": 0.88, "Cough": 0.18, "Headache": 0.70, "Body_Pain": 0.52},
    "Cold": {"Fever": 0.22, "Cough": 0.86, "Headache": 0.30, "Body_Pain": 0.20},
}

CLASS_SAMPLE_ROWS = [
    [25, "Male", "Yes", "No", "Yes", "Yes", "Malaria"],
    [14, "Female", "Yes", "Yes", "No", "No", "Flu"],
    [40, "Male", "No", "Yes", "No", "No", "Cold"],
    [32, "Female", "Yes", "Yes", "Yes", "Yes", "Typhoid"],
]


def yes_no(probability):
    return "Yes" if random.random() < probability else "No"


def generate_dataset(records=6000):
    random.seed(RANDOM_SEED)
    rows = [
        {
            "Age": age,
            "Gender": gender,
            "Fever": fever,
            "Cough": cough,
            "Headache": headache,
            "Body_Pain": body_pain,
            "Disease": disease,
        }
        for age, gender, fever, cough, headache, body_pain, disease in CLASS_SAMPLE_ROWS
    ]

    for _ in range(records - len(rows)):
        # Choose the disease first, then create symptoms that are likely for it.
        disease = random.choice(DISEASE_CLASSES)
        profile = SYMPTOM_PROFILES[disease]

        rows.append(
            {
                "Age": random.randint(1, 90),
                "Gender": random.choice(["Male", "Female"]),
                "Fever": yes_no(profile["Fever"]),
                "Cough": yes_no(profile["Cough"]),
                "Headache": yes_no(profile["Headache"]),
                "Body_Pain": yes_no(profile["Body_Pain"]),
                "Disease": disease,
            }
        )

    return pd.DataFrame(rows)


def encode_columns(data):
    encoded_data = data.copy()
    encoders = {}

    # Label Encoding changes text values like Yes/No into numbers for the model.
    for column in ["Gender", "Fever", "Cough", "Headache", "Body_Pain", "Disease"]:
        encoder = LabelEncoder()
        encoded_data[column] = encoder.fit_transform(encoded_data[column])
        encoders[column] = encoder

    return encoded_data, encoders


def train_model(records=6000):
    dataset = generate_dataset(records=records)
    dataset.to_csv(DATASET_PATH, index=False)

    encoded_dataset, encoders = encode_columns(dataset)
    x = encoded_dataset[FEATURE_COLUMNS]
    y = encoded_dataset[TARGET_COLUMN]

    # 80% of the data is used for training and 20% is used for testing.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=encoders[TARGET_COLUMN].classes_,
    )

    bundle = {
        "model": model,
        "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "accuracy": accuracy,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(bundle, BUNDLE_PATH)

    METRICS_PATH.write_text(
        f"Decision Tree Classifier Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    print(f"Dataset saved to: {DATASET_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")
    print(f"Model bundle saved to: {BUNDLE_PATH}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_model()
