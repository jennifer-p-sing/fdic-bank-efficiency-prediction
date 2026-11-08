"""FDIC XGBoost inference preprocessing

Usage
-----
    from preprocessing import preprocess_data
    X, metadata, target = preprocess_data(curated_fdic_dataframe)

Important: provide data with the same upstream FDIC definitions, periods and
units as the model-training data. Some of the model's 47 features (e.g., NIMY,
ASTEMPM, EQV) are supplied by the upstream curated FDIC dataset, not computed
by this file. 
"""

import numpy as np
import pandas as pd

EPSILON = 1e-8 # smoothing. to avoid dividing by zero incase the value is zero

# Step 1  Feature Engineered definitions of these can be found if the Feature Engineering Doc in our repository
RATIO_DEFINITIONS = [
    ("securities_to_assets", "SC", "ASSET"),
    ("cash_to_assets", "CHBAL", "ASSET"),
    ("premises_to_assets", "BKPREM", "ASSET"),
    ("intangibles_to_assets", "INTAN", "ASSET"),
    ("trading_assets_to_assets", "TRADE", "ASSET"),
    ("other_real_estate_to_assets", "ORE", "ASSET"),
    ("equity_to_assets", "EQTOT", "ASSET"),
    ("core_deposit_share", "COREDEP", "DEP"),
    ("uninsured_deposit_share", "DEPUNINS", "DEP"),
    ("demand_deposit_share", "DEPNIDOM", "DEP"),
    ("repo_funding_to_assets", "FREPO", "ASSET"),
    ("resale_agreement_to_assets", "TFRA", "ASSET"),
    ("liability_to_assets", "LIAB", "ASSET"),
    ("noncurrent_loan_ratio", "NALTOT", "LNLSNET"),
]

# Drop raw constituent columns that have been successfully replaced by ratios
RAW_CONSTITUENTS_TO_DROP = [
    "ASSET", "NUMEMP", "DEP", "LNLSNET", "SC", "CHBAL", "BKPREM",
    "INTAN", "TRADE", "ORE", "AOA", "EQ", "EQTOT", "COREDEP",
    "DEPUNINS", "DEPNIDOM", "FREPO", "TFRA", "LIAB", "NALTOT",
]


def engineer_fdic_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Apply the same ratio/log transformations as the training notebook."""
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input must be a pandas DataFrame")

    df_eng = df_input.copy()
    for output, numerator, denominator in RATIO_DEFINITIONS:
        if numerator in df_eng.columns and denominator in df_eng.columns:
            df_eng[output] = df_eng[numerator] / (df_eng[denominator] + EPSILON)

    # Step 1.5 log the assets -> Reduces skewness | clip lower ensures anything less than 1 becomes 1 before logging.
    if "ASSET" in df_eng.columns:
        df_eng["log_assets"] = np.log(df_eng["ASSET"].clip(lower=1))

    return df_eng.drop(columns=RAW_CONSTITUENTS_TO_DROP, errors="ignore")


# Step 2 Correlation/leakage reductions: features audited as redundant (r >= 0.95) or leakage risks.
CORRELATED_OR_LEAKAGE_COLUMNS = [
   'IDT1CER',                # Redundant with IDT1RWAJR (Tier 1 Capita Indicator)
   'RBC1AAJ',                # Redundant with EQV (Leverage/Equity)
   'DEPCSBQ',                # Overlapping deposit sub-component DEPDOM
   'DEPIDOM',                # Overlapping deposit sub-component DEPDOM
   'DEPBEFEX',               # Overlapping deposit sub-component DEPDOM
   'DEPINS',                 # Overlapping deposit sub-component DEPDOM
   'EEFFR',                  # avoid leakage
]

# Step 3: Unstable variables removed (missing 33% of the values)
UNSTABLE_COLUMNS = ["P3LTOT", "P9LTOT", "noncurrent_loan_ratio"]

# Step 4: Remove features with training-time XGBoost importance below 0.001.
# These scores are the fixed historical feature-importance results from training.
# Do NOT recalculate feature importance on incoming inference data.
# IMPORTANCE_THRESHOLD = 0.001

LOW_IMPORTANCE_COLUMNS = [
    "ELNANTR",                  # 0.000953
    "P3ASSET",                  # 0.000948
    "other_real_estate_to_assets",  # 0.000938
    "intangibles_to_assets",   # 0.000938
    "DEPALLEX",                 # 0.000916
    "NTRER",                    # 0.000871
    "LNRESNCR",                 # 0.000857
    "LNNDEPD",                  # 0.000837
    "IGLSEC",                   # 0.000811
    "trading_assets_to_assets", # 0.000710
    "EXTRA",                    # 0.000428
]

# Metadata and target are separated before feature engineering so they remain
# available for identifying predictions and evaluating historical records.
ID_COLUMNS = ["CERT", "NAME", "NAMEFULL", "period", "year", "quarter", "SPECGRPDESC"]
TARGET_COLUMN = "TARGET_EEFFR_NEXT_Q1"

# 47 feature columns used to fit XGBoost_model.json. these are their exact order.
FINAL_MODEL_FEATURES = [
    "NCLNLSR", "log_assets", "core_deposit_share", "NIMY", "IDNCCIR",
    "ASTEMPM", "ITAX", "EQCDIV", "EQV", "INTEXPY",
    "demand_deposit_share", "INTINC", "premises_to_assets", "IDERNCVR",
    "NFAA", "liability_to_assets", "IDNCOTHR", "IDNCCONR", "AOAR",
    "cash_to_assets", "IDLNCORR", "ERNASTR", "LNATRESR",
    "resale_agreement_to_assets", "UC", "uninsured_deposit_share",
    "IDT1RWAJR", "ELNATRY", "NAASSET", "DEPDASTR", "EQCDIVNTINC",
    "EINTEXP", "repo_funding_to_assets", "RBCRWAJ", "LNLSDEPR",
    "LNLSNTV", "P9ASSET", "NTLNLSR", "INTINCY",
    "securities_to_assets", "NTLNLS", "FLAG_NONTRADITIONAL_MODEL",
    "DEPDOM", "NPERFV", "ELNATR", "OBSDIR", "NCRER",
]


def preprocess_data(df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return model features, metadata/IDs, and target as three DataFrames.

    Parameters
    ----------
    df_input : pandas.DataFrame
        Incoming FDIC data with the same upstream curated columns, units and
        definitions as the training notebook's input.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        1st df X : with 47 numerical features in the saved model's training order.
        2nd Metadata: Available identifier columns from ID_COLUMNS, in listed order.
        3rd target: TARGET_EEFFR_NEXT_Q1 if supplied; otherwise an empty DataFrame
            with the input's index (for future inference where labels are unknown).
        The three outputs preserve input row order/index, and input is not modified.

    Raises
    ------
    ValueError
        A required model feature is absent after the engineering steps.
    TypeError
        Input is not a DataFrame or a required feature is nonnumeric.

    No missing-value imputation or additional scaling is performed, because
    those transformations were not specified in the training notebook.
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input must be a pandas DataFrame")
    if not df_input.columns.is_unique:
        raise ValueError("Input DataFrame contains duplicate column names")

    # Step 0: Preserve available metadata and the next-quarter target separately.
    id_cols = [col for col in ID_COLUMNS if col in df_input.columns]
    metadata = df_input.loc[:, id_cols].copy()
    target = (df_input.loc[:, [TARGET_COLUMN]].copy()
              if TARGET_COLUMN in df_input.columns
              else pd.DataFrame(index=df_input.index))

    # Step 1: Create engineered financial ratios and log_assets; drop constituents.
    df = engineer_fdic_features(df_input)

    # Step 2: Drop the notebook's manually identified redundant/leaky variables.
    df = df.drop(columns=CORRELATED_OR_LEAKAGE_COLUMNS, errors="ignore")

    # Step 2.5: Convert the nontraditional-model flag from bool to int
    if "FLAG_NONTRADITIONAL_MODEL" in df.columns:
        df["FLAG_NONTRADITIONAL_MODEL"] = df["FLAG_NONTRADITIONAL_MODEL"].astype(int)

    # Step 3: Drop the manually audited unstable variables.
    df = df.drop(columns=UNSTABLE_COLUMNS, errors="ignore")

    # Step 4: Apply the fixed importance-threshold exclusions from training.
    df = df.drop(columns=LOW_IMPORTANCE_COLUMNS, errors="ignore")

    # Step 5: Validate, select and order the trained model's 47 features.
    missing = [name for name in FINAL_MODEL_FEATURES if name not in df.columns]
    if missing:
        raise ValueError(
            "Missing XGBoost model features after preprocessing: "
            + ", ".join(missing)
            + ". Supply the same upstream curated FDIC columns as training, "
              "including the raw ratio inputs."
        )

    X = df.loc[:, FINAL_MODEL_FEATURES].copy()
    nonnumeric = [name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
    if nonnumeric:
        raise TypeError(
            "Model features must be numerical: " + ", ".join(nonnumeric)
        )
    return X, metadata, target
