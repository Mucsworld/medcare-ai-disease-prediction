# MedCare - AI Disease Prediction System

MedCare is a supervised machine learning project prepared for Queens Hospital Akure. The aim is to support early patient screening by predicting a possible illness from basic symptoms such as fever, cough, headache, and body pain.

This project is written as a simple, explainable classroom prototype. It is not meant to replace doctors or nurses. It is only a support tool that can help staff decide which patients may need attention quickly.

## Example Dataset Format

The dataset follows the same simple table style used in class:

| Age | Gender | Fever | Cough | Headache | Body Pain | Disease |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | Male | Yes | No | Yes | Yes | Malaria |
| 14 | Female | Yes | Yes | No | No | Flu |
| 40 | Male | No | Yes | No | No | Cold |
| 32 | Female | Yes | Yes | Yes | Yes | Typhoid |

In the CSV file, `Body Pain` is written as `Body_Pain` because spaces in column names can make Python code harder to work with.

## Project Files

- `disease_dataset.csv` - synthetic healthcare dataset with at least 5,000 records
- `train_model.py` - generates the dataset, preprocesses data, trains the model, evaluates accuracy, and saves artifacts
- `app.py` - Streamlit web application for disease prediction
- `disease_prediction_model.joblib` - trained Decision Tree model
- `label_encoders.joblib` - saved label encoders for categorical columns
- `model_bundle.joblib` - combined model, encoders, feature names, and accuracy
- `model_metrics.txt` - model accuracy and classification report
- `requirements.txt` - required Python packages

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Command note: this installs Pandas, Scikit-learn, Joblib, and Streamlit.

Start the Streamlit application:

```bash
streamlit run app.py
```

Command note: this opens the web application in the browser. If the dataset or model files are missing, the app will create them automatically.

Optional: train the model from the terminal:

```bash
python train_model.py
```

Command note: this creates `disease_dataset.csv`, trains the Decision Tree model, checks accuracy, and saves the model files without opening the Streamlit app.

## GitHub and Streamlit Deployment

To create a GitHub URL, upload the project files to a public GitHub repository named:

```text
medcare-ai-disease-prediction
```

The GitHub URL will look like:

```text
https://github.com/YOUR-USERNAME/medcare-ai-disease-prediction
```

For Streamlit Cloud deployment:

- Repository: `medcare-ai-disease-prediction`
- Main file path: `app.py`
- Requirements file: `requirements.txt`

More detailed steps are available in `GITHUB_DEPLOYMENT.md`.

## Example Prediction

Patient information:

- Age: 30
- Gender: Male
- Fever: Yes
- Cough: No
- Headache: Yes
- Body Pain: Yes

Expected model output:

```text
Predicted Disease: Malaria
```

## Notes

This project is an educational decision-support prototype. Any prediction from the system should be confirmed by qualified medical staff before treatment is given.
