# Musanze Harvest Intelligence Service

**Full Name:** Cedrick Manzi  
**Registration Number:** 25/27196  
**Institutional Username:** ug2527196@ines.ac.rw  
**Verification Code:** AI-A2-7196

## 1. Problem Description

Musanze Harvest Intelligence is a Python-based machine learning project designed to analyze agricultural data and support crop-yield estimation, high-risk prediction, and the identification of groups of similar agricultural observations.

The system includes regression, classification, and clustering models and provides a FastAPI interface for accessing predictions.

**Dataset notice:** The dataset used in this assignment is synthetic. It is not a collection of verified real-world agricultural observations.

## 2. System Requirements

- Python 3.14.7
- Windows operating system with PowerShell
- pip for Python package installation
- Git for version control

## 3. Setup Instructions

Open PowerShell and navigate to the project directory:

```powershell
cd C:\Users\user\Desktop\AI_A2_2527196
```

Create and activate the virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 4. Running the Application

Start the FastAPI application:

```powershell
python -m uvicorn app:app --reload
```

Open the interactive API documentation in your browser:

http://127.0.0.1:8000/docs

Run the automated tests:

```powershell
python -m pytest -v
```

## 5. API Usage

The API provides health-check, model-information, and prediction functionality.

Example health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example prediction request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"rainfall_mm":900,"avg_temp_c":20,"soil_moisture_pct":50,"fertilizer_kg":100,"field_size_ha":1.5,"season":"A","cooperative_support":1}'
```

Check the exact endpoint paths and required request fields in `/docs` before using these examples.

## 6. Project Artifacts

The project includes:

- `data/AI_A2_2527196.csv` — agricultural dataset.
- `artifacts/data_fingerprint.txt` — dataset fingerprint information.
- `artifacts/regression_metrics.json` — regression evaluation results.
- `artifacts/classification_metrics.json` — classification evaluation results.
- `artifacts/clustering_report.json` — clustering evaluation information.
- `models/` — saved trained models and preprocessing artifacts.
- `src/` — source code for validation, preprocessing, prediction, and machine learning.
- `tests/` — automated tests.
- `requirements.txt` — Python dependencies.
- `app.py` — FastAPI application.
- `README.md` — project documentation.

## 7. Testing and Validation

The project previously passed four automated tests covering the health endpoint, model information, prediction functionality, and rejection of an invalid season value.

The API endpoints were also tested successfully during development.

## 8. Repository Information

**GitHub repository:** https://github.com/cedrickmanzi3/AI_A2_2527196.git

**Final commit hash:** ad8ad1eb5689c576c444512a80a9e413ad3a2f35

## 9. Known Limitations

- The dataset is synthetic and may not represent real agricultural conditions in Musanze.
- Model predictions are estimates and should not be treated as guaranteed crop yields or definitive agricultural advice.
- Model performance depends on the quality and representativeness of the input data.
- Clustering labels identify groups of similar observations; they are not risk levels.
- Further testing with verified real-world agricultural data would be needed before practical deployment.

## 10. Verification

**Assignment verification code:** AI-A2-7196
