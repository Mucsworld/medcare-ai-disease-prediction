# GlycoAID - Diabetes Risk Prediction System

GlycoAID is a supervised machine learning capstone project that predicts diabetes risk from routine diagnostic health measurements.

## Dataset

The training script loads the dataset directly from the mandatory GitHub URL:

https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

No local dataset file is required.

## Project Files

- `train_model.py` - fetches the dataset, cleans invalid values, performs EDA, trains Logistic Regression and Decision Tree models, compares metrics, and saves `model.pkl`.
- `app.py` - Streamlit web application for real-time diabetes risk prediction.
- `requirements.txt` - packages needed for training and deployment.
- `model.pkl` - generated after running `train_model.py`.
- `outputs/` - generated EDA summaries, model comparison table, metadata, and charts.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

If `model.pkl` is missing, the Streamlit app will attempt to train the model automatically from the same mandatory GitHub dataset URL.

## Workflow Covered

- Loads dataset directly from GitHub.
- Handles invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
- Performs exploratory data analysis.
- Splits data into training and testing sets.
- Trains Logistic Regression and Decision Tree Classifier.
- Compares models using accuracy, precision, recall, F1 score, and ROC AUC.
- Saves the best model with Joblib as `model.pkl`.
- Provides a Streamlit UI with 8 patient inputs, real-time prediction, risk probability, model comparison, feature importance, and a brief prediction explanation.

## GitHub URL

GitHub repository:

https://github.com/Mucsworld/medcare-ai-disease-prediction

Prepared deployment branch:

```text
codex/glycoaid-diabetes
```

## Streamlit Cloud Deployment

1. Go to https://share.streamlit.io/.
2. Select **New app**.
3. Choose repository: `Mucsworld/medcare-ai-disease-prediction`.
4. Choose branch: `codex/glycoaid-diabetes`.
5. Set main file path: `app.py`.
6. Select **Deploy**.
7. Copy the live app URL for submission.

The app trains `model.pkl` automatically on first launch if the file is missing. This keeps the deployment compatible with the project rule that the dataset must be loaded directly from the GitHub URL.

## Submission Checklist

- `train_model.py`
- `app.py`
- `model.pkl`
- Screenshots of the app in the `screenshots/` folder
- Live Streamlit Cloud link
