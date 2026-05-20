# GlycoAID - Diabetes Risk Prediction System

GlycoAID is a supervised machine learning capstone project that predicts diabetes risk from routine diagnostic health measurements.

## Dataset

The dataset is loaded directly from the mandatory GitHub URL:

```text
https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
```

No local dataset file is required.

## Files

- `train_model.py` - loads the dataset from GitHub, handles invalid zero values, performs EDA, trains Logistic Regression and Decision Tree models, compares performance, and saves `model.pkl`.
- `app.py` - Streamlit web application for real-time diabetes risk prediction.
- `requirements.txt` - Python packages needed for training and deployment.
- `runtime.txt` - Python runtime for Streamlit Cloud.
- `.streamlit/config.toml` - Streamlit theme settings.
- `GITHUB_DEPLOYMENT.md` - deployment steps.
- `RUN_WINDOWS.md` - Windows/VS Code run commands.

## How To Run On Windows

Open PowerShell in the project folder:

```powershell
cd "C:\Users\dell\Documents\Codex\2026-05-20\please-read-this-fetch-the-data"
```

Install dependencies using your Python 3.13 interpreter:

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
```

Train the model:

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" train_model.py
```

Run the Streamlit app:

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

## Workflow Covered

- Loads dataset directly from GitHub.
- Handles invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
- Performs exploratory data analysis.
- Splits data into training and testing sets.
- Trains Logistic Regression and Decision Tree Classifier.
- Compares models using accuracy, precision, recall, F1 score, and ROC AUC.
- Saves the best model with Joblib as `model.pkl`.
- Provides a Streamlit UI with 8 patient inputs, real-time prediction, probability score, model comparison, feature importance, and explanation.

## Streamlit Cloud Deployment

Use these settings:

- Repository: `Mucsworld/medcare-ai-disease-prediction`
- Branch: `codex/glycoaid-diabetes`
- Main file path: `app.py`

The app automatically trains `model.pkl` on first launch if the model file is missing.

## Submission Checklist

- `train_model.py`
- `app.py`
- `model.pkl` after training
- Screenshots of the app
- Live Streamlit Cloud link
