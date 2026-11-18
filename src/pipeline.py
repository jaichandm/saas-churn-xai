import pandas as pd

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def prepare_inference_features(raw_record: dict) -> pd.DataFrame:
    """Applies identical domain feature engineering used during training."""
    df = pd.DataFrame([raw_record])

    # 1. ChargeRatio
    df["ChargeRatio"] = df["TotalCharges"] / (
        df["tenure"] * df["MonthlyCharges"] + 1e-5
    )

    # 2. ServiceBreadth
    auxiliary_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["ServiceBreadth"] = df[auxiliary_services].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # 3. HighRiskCombo
    df["HighRiskCombo"] = (
        (df["Contract"] == "Month-to-month")
        & (df["InternetService"] == "Fiber optic")
        & (df["TechSupport"] == "No")
    ).astype(int)

    # 4. ContractRiskScore
    contract_risk_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["ContractRiskScore"] = df["Contract"].map(contract_risk_map).fillna(2)

    # Ensure categorical types match training schema
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)

    return df