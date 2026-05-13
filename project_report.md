# MedCare AI Disease Prediction System - Short Report

## Introduction

Queens Hospital Akure plans to use artificial intelligence to support healthcare workers and reduce delays in patient response. Many patients ignore early symptoms or delay hospital visits because of waiting time and consultation cost. This project presents a simple AI prototype that can help staff screen patients earlier using common symptoms.

## Problem Type

This is a supervised learning classification problem. The model learns from labelled examples where each patient record contains input features and a known disease class.

## Dataset

A custom synthetic dataset was generated with 6,000 records. The table format follows the example used in class:

| Age | Gender | Fever | Cough | Headache | Body Pain | Disease |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | Male | Yes | No | Yes | Yes | Malaria |
| 14 | Female | Yes | Yes | No | No | Flu |
| 40 | Male | No | Yes | No | No | Cold |
| 32 | Female | Yes | Yes | Yes | Yes | Typhoid |

The final CSV dataset contains these columns:

- Age
- Gender
- Fever
- Cough
- Headache
- Body_Pain
- Disease

The disease classes are Malaria, Flu, Typhoid, and Cold.

## Data Preprocessing

Categorical columns such as Gender, Fever, Cough, Headache, Body_Pain, and Disease were converted into numerical values using Label Encoding. This is necessary because the Decision Tree model works with numbers, not raw text.

## Model

A Decision Tree Classifier from Scikit-learn was used because it is simple, interpretable, and suitable for an introductory supervised classification task. A decision tree is also easy to explain because it makes predictions by following symptom-based decision rules.

## Evaluation

The dataset was split into training and testing sets using an 80:20 ratio. Model performance was evaluated with accuracy score, and a classification report was saved in `model_metrics.txt`.

## Deployment

A Streamlit application was developed in `app.py`. Hospital staff can enter patient information and receive a predicted disease from the trained model.

## Command Notes

Install dependencies:

```bash
pip install -r requirements.txt
```

This command installs the Python libraries needed for the project.

Run the web app:

```bash
streamlit run app.py
```

This command starts the Streamlit application in the browser. The app can also generate the dataset and train the model automatically if the saved model files are missing.

Optional terminal training:

```bash
python train_model.py
```

This command generates the dataset, trains the Decision Tree model, evaluates accuracy, and saves the model files without opening the web app.

## Conclusion

The MedCare prototype shows how machine learning can support faster healthcare triage. The system should be treated as a support tool only. Final diagnosis and treatment decisions must remain with qualified healthcare professionals.
