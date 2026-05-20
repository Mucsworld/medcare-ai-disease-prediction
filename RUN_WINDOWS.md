# Running GlycoAID On Windows

Use these commands in PowerShell.

## 1. Go To The Project Folder

```powershell
cd "C:\Users\dell\Documents\Codex\2026-05-20\please-read-this-fetch-the-data"
```

## 2. Install Required Packages

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
```

If `pip` is missing, run:

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m ensurepip --upgrade
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m pip install --upgrade pip
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
```

## 3. Train The Model

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" train_model.py
```

This creates:

- `model.pkl`
- `outputs/model_comparison.csv`
- `outputs/model_metadata.json`
- EDA charts inside `outputs/figures/`

## 4. Run The Streamlit App

```powershell
& "C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

## Common Error

If you see:

```text
ModuleNotFoundError: No module named 'joblib'
```

It means dependencies were not installed for the Python interpreter you used. Run step 2 again.
