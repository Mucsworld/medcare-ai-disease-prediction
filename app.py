import random
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MedCare Disease Prediction",
    layout="centered",
)

try:
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier
except ModuleNotFoundError:
    st.title("MedCare Disease Prediction System")
    st.error("Scikit-learn is not installed in this Python environment.")
    st.write("Open a terminal in the project folder and run:")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.stop()


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


st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 14px 16px;
        background: #f8fafc;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def train_and_save_model(records=6000):
    dataset = generate_dataset(records=records)
    dataset.to_csv(DATASET_PATH, index=False)

    encoded_dataset, encoders = encode_columns(dataset)
    x = encoded_dataset[FEATURE_COLUMNS]
    y = encoded_dataset[TARGET_COLUMN]

    # 80% of the data trains the model, 20% tests the model accuracy.
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

    return bundle, dataset


@st.cache_resource
def load_or_prepare_model():
    if BUNDLE_PATH.exists() and DATASET_PATH.exists():
        dataset = pd.read_csv(DATASET_PATH)
        return joblib.load(BUNDLE_PATH), dataset

    return train_and_save_model(records=6000)


def encode_patient_data(patient_data, encoders):
    encoded = patient_data.copy()

    for column in ["Gender", "Fever", "Cough", "Headache", "Body_Pain"]:
        encoded[column] = encoders[column].transform([encoded[column]])[0]

    return pd.DataFrame([encoded], columns=FEATURE_COLUMNS)


def predict_disease(patient_data, bundle):
    encoded_input = encode_patient_data(patient_data, bundle["encoders"])
    prediction = bundle["model"].predict(encoded_input)[0]
    return bundle["encoders"]["Disease"].inverse_transform([prediction])[0]


bundle, dataset = load_or_prepare_model()

st.title("MedCare Disease Prediction System")
st.caption("Queens Hospital Akure patient-support prototype")

left_metric, right_metric = st.columns(2)
left_metric.metric("Dataset Records", f"{len(dataset):,}")
right_metric.metric("Model Accuracy", f"{bundle['accuracy'] * 100:.2f}%")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        fever = st.selectbox("Fever", ["Yes", "No"])

    with right:
        cough = st.selectbox("Cough", ["Yes", "No"])
        headache = st.selectbox("Headache", ["Yes", "No"])
        body_pain = st.selectbox("Body Pain", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Disease")

if submitted:
    patient = {
        "Age": age,
        "Gender": gender,
        "Fever": fever,
        "Cough": cough,
        "Headache": headache,
        "Body_Pain": body_pain,
    }

    disease = predict_disease(patient, bundle)

    st.success(f"Predicted Disease: {disease}")
    st.info(
        "Please review this result with a qualified healthcare worker before making any medical decision."
    )

st.divider()
st.subheader("Example Dataset")
st.dataframe(dataset.head(4), use_container_width=True)

with st.expander("Command notes"):
    st.code("pip install -r requirements.txt", language="bash")
    st.write("Installs Pandas, Scikit-learn, Joblib, and Streamlit.")
    st.code("streamlit run app.py", language="bash")
    st.write("Runs this app. The app will create the dataset and model files automatically if they are missing.")
