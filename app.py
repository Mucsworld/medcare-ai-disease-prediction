"""Streamlit app for GlycoAID diabetes risk prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


MODEL_PATH = Path("model.pkl")
METADATA_PATH = Path("outputs/model_metadata.json")
COMPARISON_PATH = Path("outputs/model_comparison.csv")
IMPORTANCE_PATH = Path("outputs/feature_importance.csv")

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


st.set_page_config(
    page_title="GlycoAID Diabetes Risk Prediction",
    page_icon="G",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8f7;
        color: #17211f;
    }
    .hero {
        padding: 1.4rem 0 0.8rem 0;
        border-bottom: 1px solid #d9e4e0;
        margin-bottom: 1.1rem;
    }
    .hero h1 {
        font-size: 2.25rem;
        line-height: 1.1;
        margin: 0;
        color: #123b35;
    }
    .hero p {
        max-width: 780px;
        color: #50615d;
        font-size: 1rem;
        margin-top: 0.45rem;
    }
    .result-low, .result-high {
        border-radius: 8px;
        padding: 1rem 1.1rem;
        border: 1px solid;
        margin-top: 0.35rem;
    }
    .result-low {
        background: #eaf6ef;
        border-color: #9fd0b2;
        color: #155b31;
    }
    .result-high {
        background: #fff0ed;
        border-color: #eca399;
        color: #8a2118;
    }
    [data-testid="stMetricValue"] {
        color: #123b35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def ensure_model_file() -> None:
    if MODEL_PATH.exists():
        return

    with st.spinner("Training model from the mandatory GitHub dataset..."):
        from train_model import main as train_model_main

        train_model_main()


def prepare_patient_frame(values: dict[str, float]) -> pd.DataFrame:
    patient = pd.DataFrame([values], columns=FEATURES)
    patient[ZERO_AS_MISSING] = patient[ZERO_AS_MISSING].replace(0, np.nan)
    return patient


def probability_gauge(probability: float) -> go.Figure:
    percent = probability * 100
    color = "#c7504a" if probability >= 0.5 else "#2e7d6b"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Diabetes Risk Probability"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#d9e4e0",
                "steps": [
                    {"range": [0, 50], "color": "#eaf6ef"},
                    {"range": [50, 100], "color": "#fff0ed"},
                ],
                "threshold": {
                    "line": {"color": "#17211f", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor="#f6f8f7")
    return fig


def render_explanation(patient: pd.DataFrame, importance: pd.DataFrame) -> None:
    if importance.empty:
        return

    top_features = importance.head(3)["feature"].tolist()
    st.subheader("Why This Prediction?")
    st.write(
        "The strongest signals for this model are "
        + ", ".join(top_features)
        + ". The patient values for those signals are shown below."
    )
    st.dataframe(patient[top_features].T.rename(columns={0: "Patient Value"}), use_container_width=True)


st.markdown(
    """
    <div class="hero">
      <h1>GlycoAID</h1>
      <p>Diabetes risk prediction from routine diagnostic health measurements.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    ensure_model_file()
except Exception as exc:
    st.error("model.pkl was not found and automatic training could not complete.")
    st.exception(exc)
    st.stop()

model = load_model()
metadata = load_metadata()
comparison = load_table(COMPARISON_PATH)
importance = load_table(IMPORTANCE_PATH)

with st.sidebar:
    st.header("Patient Inputs")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=250.0, value=120.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=140.0, value=72.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=32.0, step=0.1)
    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.45,
        step=0.01,
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
    predict_button = st.button("Predict Risk", type="primary", use_container_width=True)

patient_values = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": pedigree,
    "Age": age,
}
patient = prepare_patient_frame(patient_values)

left, right = st.columns([1, 1])

with left:
    st.subheader("Patient Snapshot")
    display_patient = pd.DataFrame(
        {
            "Feature": [
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "Skin Thickness",
                "Insulin",
                "BMI",
                "Diabetes Pedigree Function",
                "Age",
            ],
            "Value": list(patient_values.values()),
        }
    )
    st.dataframe(display_patient, hide_index=True, use_container_width=True)

    if metadata:
        st.metric("Selected Model", metadata.get("selected_model", "Unknown"))

with right:
    st.subheader("Prediction")
    if predict_button:
        prediction = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0][1])

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-high">
                  <h3>High Risk of Diabetes</h3>
                  <p>Estimated risk: <strong>{probability:.0%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-low">
                  <h3>Low Risk of Diabetes</h3>
                  <p>Estimated risk: <strong>{probability:.0%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.plotly_chart(probability_gauge(probability), use_container_width=True)
    else:
        st.info("Enter patient measurements in the sidebar and select Predict Risk.")

tabs = st.tabs(["Model Comparison", "Feature Importance", "Explanation"])

with tabs[0]:
    st.subheader("Model Comparison")
    if comparison.empty:
        st.warning("Run the training script to generate the model comparison table.")
    else:
        st.dataframe(comparison.round(4), hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Feature Importance")
    if importance.empty:
        st.warning("Run the training script to generate feature importance values.")
    else:
        st.bar_chart(importance.set_index("feature")["importance"])

with tabs[2]:
    render_explanation(patient, importance)
    st.caption("This screening tool supports clinical review and is not a standalone diagnosis.")
