# 🛡️ Enterprise SaaS Churn Intelligence & Explainable AI (XAI)

**Cost-Calibrated Gradient Boosting • Native C++ TreeSHAP Attribution • Real-Time FastAPI & Streamlit Serving**

[System Architecture](#-system-architecture) •
[Notebook Suite](#-notebook-suite-what--how--why) •
[External Setup & Testing Guide](#-external-user-setup--end-to-end-testing-guide) •
[Live API Payload](#-live-inference--xai-payload)

## 📌 Executive Summary

Standard churn models fail in production because they optimize for symmetric accuracy at a default `0.50` probability cutoff—ignoring asymmetric SaaS unit economics where **losing an account (\$500 LTV) costs 10x more than proactive retention outreach ($50 incentive)**.

This repository implements an end-to-end, leakage-free Machine Learning and Explainable AI system that:

* **Aligns ML with Unit Economics:** Replaces arbitrary `0.50` thresholding with empirical loss minimization ($\text{Cost} = 500 \cdot FN + 50 \cdot FP$), shifting the operational decision boundary to **`0.095`**.

* **Eliminates Black-Box Friction:** Uses LightGBM's native C++ **TreeSHAP** (`pred_contrib=True`) to decompose every account prediction into log-odds feature drivers mapped to Customer Success playbooks.

* **Prevents Train-Serve Skew:** Enforces strict stratified splitting prior to feature engineering, encapsulates identical domain transformations in `src/pipeline.py`, and serves sub-30ms inference with zero `pandas` overhead at runtime.

## 🏗️ System Architecture

```
flowchart LR
    A[("Raw Kaggle Data<br/>(7,043 Accounts)")] --> B["Stratified 80/20 Split<br/>& Domain Engineering"]
    B --> C["ColumnTransformer<br/>(StandardScaler + OHE)"]
    C --> D["Optuna 5-Fold CV<br/>LightGBM Classifier"]
    D --> E["Financial Cost Curve<br/>Threshold Optimizer"]

    C -.-> F[("models/<br/>preprocessor.joblib")]
    D -.-> G[("models/<br/>lgbm_churn_model.txt")]
    E -.-> H[("models/<br/>threshold_config.json")]

    F & G & H --> I["ChurnPredictorService<br/>(Zero-Pandas + C++ TreeSHAP)"]
    I --> J["⚡ FastAPI REST<br/>POST /predict"]
    I --> K["📊 Streamlit Portal<br/>CS Decision Workspace"]

```

## 📂 Repository Structure

```
saas-churn-xai/
├── app/
│   ├── api.py                                        # FastAPI REST microservice (/health, /predict)
│   └── dashboard.py                                  # Interactive Streamlit Customer Success workspace
├── data/
│   ├── raw/                                          # Raw Kaggle CSV (gitignored — see setup guide)
│   └── processed/                                    # Leak-free NumPy train/test arrays (gitignored)
├── models/
│   ├── feature_names.json                            # Post-encoding feature alignment schema
│   ├── lgbm_churn_model.txt                          # Serialized Optuna-tuned LightGBM booster
│   ├── preprocessor.joblib                           # Fitted Scikit-Learn ColumnTransformer
│   └── threshold_config.json                         # Business cost matrix & optimal cutoff (0.095)
├── notebooks/
│   ├── 01_eda_and_business_metrics.ipynb             # Unit economics, MRR risk audit & cohort decay
│   ├── 02_pipeline_and_leakage_free_engineering.ipynb # Stratified split, domain features & serialization
│   ├── 03_cost_sensitive_modeling_and_tuning.ipynb   # Baselines, Optuna PR-AUC tuning & cost curve
│   └── 04_explainable_ai_shap_insights.ipynb         # Global Beeswarm, dependence & local Waterfall plots
├── src/
│   ├── __init__.py
│   ├── inference.py                                  # Scoring engine + native TreeSHAP action mapper
│   ├── pipeline.py                                   # Deterministic feature engineering (train/serve parity)
│   └── schema.py                                     # Strict Pydantic v2 request/response contracts
├── Dockerfile                                        # Production container specification
├── requirements.txt                                  # Locked dependencies
└── ReadMe.md

```

## 📓 Notebook Suite (What / How / Why)

Every modeling decision is documented across four standalone, narrative-driven notebooks:

| \# | Notebook | Core Objective | Key Engineering & Analytical Outcomes | 
| ----- | ----- | ----- | ----- | 
| **01** | [`01_eda_and_business_metrics.ipynb`](notebooks/01_eda_and_business_metrics.ipynb) | **Business Health Audit** | Remediated whitespace strings in `TotalCharges` (`tenure == 0`); quantified MRR bleed across contract tiers; uncovered steep 0–12 month survival drop-off. | 
| **02** | [`02_pipeline_and_leakage_free_engineering.ipynb`](notebooks/02_pipeline_and_leakage_free_engineering.ipynb) | **Leak-Proof Feature Pipeline** | Executed stratified 80/20 split *before* transforms; engineered `ChargeRatio`, `ServiceBreadth`, `HighRiskCombo`, and `ContractRiskScore`. | 
| **03** | [`03_cost_sensitive_modeling_and_tuning.ipynb`](notebooks/03_cost_sensitive_modeling_and_tuning.ipynb) | **Cost-Sensitive Optimization** | Benchmarked Logistic Regression, Random Forest, and LightGBM on **PR-AUC**; ran 40 Optuna 5-fold CV trials; minimized asymmetric financial loss. | 
| **04** | [`04_explainable_ai_shap_insights.ipynb`](notebooks/04_explainable_ai_shap_insights.ipynb) | **Explainable AI (SHAP)** | Generated macro Beeswarm rankings, `tenure` × `MonthlyCharges` dependence curves, local Waterfall breakdowns, and CS playbook rules. | 

## 💰 Business Unit Economics & Cost Calibration

### Asymmetric Loss Formulation

Rather than evaluating models at an arbitrary `0.50` cutoff, we calibrate the decision boundary $p^*$ directly against unit financial impact:

$$
\mathcal{L}(p) = \left(FN(p) \times \$500\right) + \left(FP(p) \times \$50\right)
$$

| Parameter | Value | Business Rationale | 
| ----- | ----- | ----- | 
| **False Negative Cost (**$C_{FN}$**)** | **`$500.00`** | Forfeited Annual Contract Value / Customer Lifetime Value from an unflagged churner | 
| **False Positive Cost (**$C_{FP}$**)** | **`$50.00`** | Proactive retention discount or Customer Success outreach overhead | 
| **Imbalance Weighting** | `scale_pos_weight` | Corrects \~26.5% minority class skew during gradient boosting | 
| **Calibrated Operating Cutoff (**$p^*$**)** | **`0.095`** | Minimizes total expected test-set revenue erosion under a 10:1 FN/FP penalty ratio | 

### Engineered Domain Features (`src/pipeline.py`)

* **`ChargeRatio`**: $\frac{\text{TotalCharges}}{\text{tenure} \times \text{MonthlyCharges} + 10^{-5}}$ — Captures billing acceleration/volatility while guarding against `tenure == 0` division.

* **`ServiceBreadth`**: Count (`0–6`) of active sticky add-ons (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).

* **`HighRiskCombo`**: Binary interaction flag isolating the highest-churn segment (`Month-to-month` + `Fiber optic` + `TechSupport == No`).

* **`ContractRiskScore`**: Ordinal commitment vulnerability (`Month-to-month: 2`, `One year: 1`, `Two year: 0`).

## 🧑‍💻 External User Setup & End-to-End Testing Guide

Follow these steps to clone, reproduce the training pipeline from scratch, and verify both serving interfaces on your own machine.

### Prerequisites

* **Python:** `3.11+` recommended (note: on Windows 11 with **Smart App Control** enabled, Python `3.14` local virtual environments may block unsigned C-extensions in `pandas`/`shap`—use **WSL2 (Ubuntu)** or **Docker** if you encounter `ImportError: DLL load failed` on Windows).

* **Kaggle Account:** Required to download the raw Telco Customer Churn CSV (`blastchar/telco-customer-churn`).

### Step 1: Clone the Repository & Create a Virtual Environment

```
git clone https://github.com/jaichandm/saas-churn-xai.git
cd saas-churn-xai

# Create and activate virtual environment
python -m venv venv

# Linux / macOS / WSL2:
source venv/bin/activate

# Windows PowerShell:
# .\venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

### Step 2: Download Raw Dataset & Generate Model Artifacts

Because raw datasets and binary model weights are excluded via `.gitignore`, regenerate them cleanly by running the notebooks in sequence:

1. **Download Dataset:**

   * Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle (`blastchar/telco-customer-churn`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

   * Place the file inside `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
     *(Tip: If you have the Kaggle CLI configured, run `kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip`)*

2. **Run Notebooks in Order (`01` → `04`):**

   ```
   jupyter notebook
   
   ```

   Execute all cells in:

   * `notebooks/01_eda_and_business_metrics.ipynb` *(verifies data integrity & MRR metrics)*

   * `notebooks/02_pipeline_and_leakage_free_engineering.ipynb` *(generates `data/processed/*.npy`, `models/preprocessor.joblib`, and `models/feature_names.json`)*

   * `notebooks/03_cost_sensitive_modeling_and_tuning.ipynb` *(runs Optuna tuning and generates `models/lgbm_churn_model.txt` and `models/threshold_config.json`)*

   * `notebooks/04_explainable_ai_shap_insights.ipynb` *(validates global & local SHAP explanations)*

3. **Verify Generated Artifacts:**

   ```
   ls -lh models/
   # Expected files:
   # - feature_names.json
   # - lgbm_churn_model.txt
   # - preprocessor.joblib
   # - threshold_config.json
   
   ```

### Step 3: Start & Test the FastAPI Inference Service

1. **Launch the API Server:**

   ```
   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
   
   ```

2. **Test Health Endpoint:**

   ```
   curl http://127.0.0.1:8000/health
   # Expected: {"status":"healthy","optimal_threshold":0.095}
   
   ```

3. **Test Live Prediction (`POST /predict`):**
   Open **`http://127.0.0.1:8000/docs`** in your browser for interactive Swagger UI testing, or send a request via `curl`:

   ```
   curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Male",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "tenure": 4,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 89.5,
       "TotalCharges": 358.0
     }'
   
   ```

4. **Test Schema Validation (`422 Unprocessable Entity`):**
   Send an invalid `Contract` value (e.g., `"Weekly"`) or negative `tenure` (`-5`) to confirm Pydantic v2 rejects malformed inputs before inference.

### Step 4: Start & Test the Streamlit Customer Success Workspace

In a new terminal (with your virtual environment active):

```
# Linux / macOS / WSL2:
PYTHONPATH=. streamlit run app/dashboard.py

# Windows PowerShell:
$env:PYTHONPATH="."; streamlit run app/dashboard.py

```

Open **`http://localhost:8501`** and run these three verification checks in the sidebar:

| Test Scenario | Sidebar Inputs | Expected Dashboard Behavior | 
| ----- | ----- | ----- | 
| **1. High-Risk Account** | `Tenure = 4`, `Month-to-month`, `Fiber optic`, `TechSupport = No`, `Electronic check` | `Outreach Required: YES`, `Risk Tier: High`, surfaces top 3 SHAP risk drivers & 3 playbook cards | 
| **2. Loyal Account** | `Tenure = 60`, `Two year`, `DSL`, `TechSupport = Yes`, `Credit card (automatic)` | `Churn Probability` drops below `9.5%`, `Risk Tier: Low`, `Outreach Required: NO` | 
| **3. Month-0 Edge Case** | `Tenure = 0`, `TotalCharges = 0.0` | Computes `ChargeRatio` cleanly via `+ 1e-5` epsilon guard with zero errors | 

### Step 5: Alternative — One-Command Docker Deployment

Once the `models/` artifacts are generated from Step 2, you can containerize and run the FastAPI microservice without managing local Python environments:

```
docker build -t saas-churn-xai:latest .
docker run -d -p 8000:8000 --name churn-api saas-churn-xai:latest

# Verify container health
curl http://localhost:8000/health

```

## ⚡ Live Inference & XAI Payload

```
{
  "churn_probability": 0.5295,
  "decision_threshold": 0.095,
  "risk_tier": "High",
  "flag_for_outreach": true,
  "top_churn_drivers": [
    {
      "feature": "ContractRiskScore",
      "shap_impact": 0.7206
    },
    {
      "feature": "PaymentMethod_Electronic check",
      "shap_impact": 0.3984
    },
    {
      "feature": "HighRiskCombo",
      "shap_impact": 0.3624
    }
  ],
  "recommended_actions": [
    "Offer a 15% discount to migrate from Month-to-Month to a 1-Year contract.",
    "Incentivize switch to Auto-Pay (Credit Card/Bank Transfer) with a $10 credit.",
    "Assign priority technical onboarding specialist + complimentary Tech Support trial."
  ]
}

```

## 🛠️ Tech Stack & Design Choices

| Layer | Technologies | Architectural Decision | 
| ----- | ----- | ----- | 
| **Data & Pipelines** | `NumPy`, `Pandas`, `Scikit-Learn`, `Joblib` | `ColumnTransformer` prevents leakage during training; lightweight NumPy shim (`_MinimalFrame`) eliminates heavy `pandas` overhead during real-time API inference. | 
| **Modeling & Tuning** | `LightGBM`, `Optuna` | Leaf-wise GBDT optimized via Bayesian TPE search on **PR-AUC** (`average_precision`) to handle minority-class churn skew. | 
| **Explainability** | `SHAP`, LightGBM `pred_contrib=True` | Notebook 04 uses full `shap` visualizations; production API uses LightGBM's native C++ TreeSHAP engine for deterministic, sub-millisecond attribution. | 
| **Serving & Validation** | `FastAPI`, `Pydantic v2`, `Uvicorn`, `Streamlit`, `Docker` | Strict literal schema enforcement on all 19 customer attributes with interactive Swagger docs and a frontline CS simulator. | 

## 📄 Dataset Attribution

* **Source:** [Telco Customer Churn — Kaggle (`blastchar/telco-customer-churn`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

* **Records:** 7,043 customer accounts across 21 demographic, billing, and service subscription attributes.

## ⚠️ Disclaimer

* **Educational & Portfolio Purpose:** This project is created strictly for educational, research, and portfolio demonstration purposes. It is not intended to serve as a turnkey commercial product or standalone financial/business advisory tool.

* **Illustrative Financial Assumptions:** The unit economics parameters used for threshold calibration (e.g., $500 False Negative LTV loss and $50 False Positive retention cost) and the Customer Success playbook recommendations are illustrative benchmarks modeled for demonstration and must be recalibrated to reflect actual business metrics before any real-world deployment.

* **Dataset Ownership & Licensing:** All dataset rights belong to their respective original owners on Kaggle (blastchar/telco-customer-churn). Users are responsible for complying with Kaggle's terms of use and applicable data privacy regulations (such as GDPR or CCPA) when handling customer data.

* **No Warranty & Limitation of Liability:** This software and its associated models are provided "as is", without warranty of any kind, express or implied. In no event shall the author(s) be held liable for any claim, damages, revenue loss, or operational decisions arising from the use or misuse of this repository.