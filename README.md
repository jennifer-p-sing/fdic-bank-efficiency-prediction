# 🏦 Predicting U.S. Bank Operating Efficiency with FDIC Financial Data
Created by: Jennifer Sing   | Connect with me: [LinkedIn](https://linkedin.com/in/jennifer-p-sing/)  

---

**Predictive analytics · Financial feature engineering · Model explainability · Early-warning simulation**

This project investigates the drivers of U.S. bank operating efficiency and forecasts **next-year Q1 efficiency ratios** using FDIC Call Report financial data. We compare Multiple Linear Regression (MLR), Decision Tree, Random Forest, and XGBoost, then demonstrate how a preserved XGBoost model can score new bank records in a **local inference simulator**.

> 💡 **Scope:** The inference notebook is an educational, local demonstration. It is **not** a live AWS deployment or a reproduction of AWS scheduling, S3 integration, drift alerts, or operational monitoring. The presentation discusses an AWS batch inference architecture; see the report for its design.

## 🎯 Project objective

The **efficiency ratio (EEFFR)** measures operating costs relative to revenue:

| Numerator | Denominator |
|:---:|:---:|
| Non-interest expense | Net interest income + Non-interest income |

**EEFFR = (Numerator ÷ Denominator) × 100%**

A **lower** EEFFR generally indicates a leaner operating structure. Our primary questions are:

1. Which financial and operational characteristics are associated with bank efficiency?
2. Can current-year Q1 financial information predict the following year's Q1 EEFFR?
3. Can model forecasts help prioritize banks for further review when their efficiency ratio may deteriorate?

These models identify predictive associations, **not causal effects**.

## 📊 Data and evaluation design

The original project used FDIC BankFind Suite quarterly financial data, covering **Q1 2022 through Q2 2026**. For next-year Q1 forecasting, the modeling data were organized as follows:

| Dataset | Input snapshot | Prediction target | Role |
|---|---|---|---|
| Train | Q1 2022–2024 | Following-year Q1 EEFFR | Model fitting / time-aware validation |
| Test | Q1 2025 | Q1 2026 EEFFR | Held-out evaluation |
| Out of time (OOT) | Q1 2026 | Q1 2027 EEFFR (not yet observed in the project) | Inference demonstration |

The refactored training notebook reads separate CSV files instead of querying SQL or S3. Input files must contain the expected **curated FDIC schema**, not just arbitrary raw BankFind exports. The OOT dataset is an **example inference input**, not a labeled test set.

## 🧠 Model comparison

The following are the **presentation's reported test-set results**, not new measurements produced by this README. All models were evaluated on **4,287 banks**.

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Reported features |
|---|---:|---:|---:|---:|
| **XGBoost** | **9.290** | **37.204** | **0.178** | 47 |
| Random Forest | 9.415 | 37.781 | 0.152 | 25 |
| Decision Tree | 11.350 | 38.808 | 0.106 | 10 |
| Multiple Linear Regression | 12.304 | 38.915 | 0.101 | 24 |

**📌 Interpretation:** XGBoost had the lowest MAE among the four trained models. However, the project's **naïve current-EEFFR benchmark** had a lower MAE (**7.45**) on the test set. The trained models are therefore not claimed to outperform this benchmark on overall forecast accuracy. The presentation also investigates their use for **early-warning screening**.

**🔁 Reproducibility note:** The refactored CSV-based XGBoost training workflow may select **45 features** using the historical importance cutoff (`importance > 0.001`), whereas the **saved JSON model uses the original 47 features**. These are distinct model versions; do not interchange their feature schemas or assume identical predictions. The final training results may differ slightly from the presentation.

## 🔍 Explainability: SHAP analysis

![SHAP feature importance and driver impact](images/Top%2020%20Features%20Shap%20for%20Stakeholder%20Action.png)

*SHAP summary plot for the preserved XGBoost analysis. Each point represents a bank observation. Horizontal position shows the feature's contribution to the model output; color indicates feature value (blue = low, red = high). SHAP values explain model behavior, not causal effects.*

The plot highlights **noncurrent loans (`NCLNLSR`)**, **net interest margin (`NIMY`)**, **bank size (`log_assets`)**, and **assets per employee (`ASTEMPM`)** among influential predictors. The distribution of SHAP values shows that a feature's impact varies across banks; it is not a fixed marginal effect.

## 📁 Repository structure

```text
.
├── data/                       # Train, test and out-of-time CSV inputs
├── images/                     # SHAP plot and other project figures
│   └── Top 20 Features Shap for Stakeholder Action.png
├── model/                      # Preserved model artifact and feature documentation
│   ├── xgboost_model.json
│   └── XGBoost_final_features_with_descriptions.csv
├── notebooks/
│   ├── run_fdic_models.ipynb   # Train/evaluate four models from CSVs
│   └── xgboost_inference_demo.ipynb  # Local inference simulator
├── presentation-report/        
├── src/
│   ├── feature_engineering.py
│   ├── model_mlr.py
│   ├── model_decision_tree.py
│   ├── model_random_forest.py
│   ├── model_xgboost.py
│   └── preprocessing.py        # Original 47-feature inference preprocessing
├── requirements.txt
└── README.md
```

> The layout above uses **`model/` (singular)** to match the current folder structure. Update filenames in this diagram if your committed files differ. In particular, ensure the saved JSON and notebook paths agree.

## 🚀 Getting started

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m pip install -r requirements.txt
```

Replace the example repository URL with your actual GitHub URL. The supplied inference requirements specify **XGBoost 3.4.1**; a complete training environment may require additional libraries such as scikit-learn, statsmodels, matplotlib, and Jupyter. Record and test their versions before publishing a fully reproducible training environment.

**2. Prepare the data**

Place the relevant curated CSVs in `data/`:

```text
data/train.csv
data/test.csv
data/out_of_time.csv
```

Check that the filenames match those configured in the notebooks. Avoid committing confidential, restricted, or unnecessarily large datasets; a permitted sample and data dictionary can be provided instead.

**3. Run the training comparison**

Open `notebooks/run_fdic_models.ipynb` and execute the cells in order. It imports the reusable model modules from `src/` and evaluates the four trained models on the held-out test data.

**4. Try the saved XGBoost model without retraining**

Open `notebooks/xgboost_inference_demo.ipynb`. It should use:

- `model/xgboost_model.json` — the preserved **47-feature** trained model;
- `src/preprocessing.py` — the matching ratio engineering, feature exclusions, and exact ordered feature list;
- `data/out_of_time.csv` — Q1 2026 input data for forecasting Q1 2027.

The inference workflow is:

```text
Curated FDIC Q1 CSV
       ↓
src/preprocessing.py
       ↓
47 features in the saved model's exact order
       ↓
model/xgboost_model.json
       ↓
Predicted next-year Q1 EEFFR
       ↓
Compare with current-year EEFFR → early-warning flag
       ↓
Local CSV predictions / review list
```

The simulator's illustrative screening rule is:

```python
predicted_change = predicted_next_year_eeffr - current_year_eeffr
flag_for_review = predicted_change > 10
```

**Current-year EEFFR is used only for the post-prediction comparison**, not as an XGBoost input feature. The prediction target is not available for the OOT period, so no OOT accuracy score should be reported until actual Q1 2027 outcomes exist. The flags are screening aids, not findings of bank distress.

### 💻 Minimal example: load the preserved model

Assuming the repository root is the current working directory:

```python
import pandas as pd
import xgboost as xgb
from src.preprocessing import preprocess_data

raw = pd.read_csv("data/out_of_time.csv")
X, metadata, target = preprocess_data(raw)

model = xgb.XGBRegressor()
model.load_model("model/xgboost_model.json")
assert list(X.columns) == list(model.feature_names_in_)

results = metadata.copy()
results["predicted_next_q1_eeffr"] = model.predict(X)

if "EEFFR" in raw.columns:
    results["current_eeffr"] = pd.to_numeric(raw["EEFFR"], errors="coerce").to_numpy()
    results["predicted_change"] = (
        results["predicted_next_q1_eeffr"] - results["current_eeffr"]
    )
    results["flag_for_review"] = results["predicted_change"] > 10

print(results.head())
```

The example assumes the original model expects predictions on the **raw EEFFR scale**. It requires the same upstream FDIC variable definitions and units used during training. The preprocessing module intentionally raises an error if a required feature is missing, rather than silently inventing values.

## 🧪 Methods and reproducibility

- **MLR:** Financial ratios, log-transformed target, train-only winsorization, specialization encoding, and OLS analysis. Different original MLR specifications should not be conflated.
- **Decision Tree:** Data completeness and near-duplicate filtering, permutation importance, top-10 feature selection, and a log1p target.
- **Random Forest:** Mutual-information feature selection, 25 inputs, 200 estimators, and a log1p target.
- **XGBoost:** Engineered ratios, log-assets, correlation/leakage exclusions, pseudo-Huber regression, and model-specific importance selection.
- **Preserved inference model:** `src/preprocessing.py` fixes the **47 names and order** required by the JSON artifact. Its transformations are separate from the newer training refactor.

The training and inference workflows are related but **not interchangeable**. Do not pass the newer 45-feature matrix into the preserved 47-feature JSON model.

## ⚠️ Limitations and responsible use

- Data quality, missingness, extreme EEFFR observations, and changing bank conditions affect performance.
- A low MAE on the full sample is not the only consideration for an early-warning screening tool; false alerts and missed deteriorations matter.
- Feature importance and SHAP values are **associational model explanations**, not evidence that changing a feature would cause the predicted improvement.
- This repository's simulator does **not** run AWS infrastructure, PSI drift monitoring, SNS notifications, or a production monitoring service.
- The project is for **educational and analytical demonstration**, not regulatory or investment decision-making without independent validation.

## 📚 Project materials

See `presentation-report/` for the project methodology, model comparisons, explainability findings, and the proposed AWS batch inference architecture.
