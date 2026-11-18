from typing import List, Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = "Male"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(..., ge=0, le=120, description="Months with company")
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(..., gt=0, description="Current monthly bill ($)")
    TotalCharges: float = Field(..., ge=0, description="Total billed to date ($)")


class ChurnDriver(BaseModel):
    feature: str
    shap_impact: float


class PredictionResponse(BaseModel):
    churn_probability: float
    decision_threshold: float
    risk_tier: Literal["Low", "High", "Critical"]
    flag_for_outreach: bool
    top_churn_drivers: List[ChurnDriver]
    recommended_actions: List[str]