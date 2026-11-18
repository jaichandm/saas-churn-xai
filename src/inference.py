import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import shap

from src.pipeline import prepare_inference_features
from src.schema import ChurnDriver, CustomerInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

INTERVENTION_PLAYBOOK = {
    "ContractRiskScore": "Offer a 15% discount to migrate from Month-to-Month to a 1-Year contract.",
    "Contract_One year": "Reinforce annual renewal perks 60 days prior to contract expiration.",
    "Contract_Two year": "Maintain standard executive check-in cadence.",
    "tenure": "Enroll account in 90-day high-touch onboarding success track.",
    "HighRiskCombo": "Assign priority technical onboarding specialist + complimentary Tech Support trial.",
    "TechSupport_Yes": "Highlight existing support utilization in quarterly business review.",
    "InternetService_Fiber optic": "Audit line stability/latency and offer bandwidth optimization check.",
    "MonthlyCharges": "Review plan tier utilization to prevent bill shock or offer bundle credit.",
    "ChargeRatio": "Audit recent billing spikes or overage charges with account admin.",
    "OnlineSecurity_Yes": "Bundle security suite add-on to increase product stickiness.",
    "PaymentMethod_Electronic check": "Incentivize switch to Auto-Pay (Credit Card/Bank Transfer) with a $10 credit.",
}


class ChurnPredictorService:
    def __init__(self):
        self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        self.booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_churn_model.txt"))
        self.explainer = shap.TreeExplainer(self.booster)

        with open(MODELS_DIR / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        with open(MODELS_DIR / "threshold_config.json", "r") as f:
            config = json.load(f)
            self.threshold = float(config["optimal_threshold"])

    def predict(self, customer: CustomerInput) -> PredictionResponse:
        # 1. Engineer features & transform
        df_features = prepare_inference_features(customer.model_dump())
        X_proc = self.preprocessor.transform(df_features)

        # 2. Score churn probability
        prob = float(self.booster.predict(X_proc)[0])

        # 3. Compute local SHAP attribution
        shap_vals = self.explainer.shap_values(X_proc)[0]
        sorted_indices = np.argsort(shap_vals)[::-1]

        top_drivers = []
        recommendations = []

        for idx in sorted_indices:
            impact = float(shap_vals[idx])
            if impact <= 0 or len(top_drivers) >= 3:
                continue

            feat = self.feature_names[idx]
            top_drivers.append(ChurnDriver(feature=feat, shap_impact=round(impact, 4)))

            action = INTERVENTION_PLAYBOOK.get(
                feat,
                f"Schedule Customer Success review focusing on '{feat}' configuration.",
            )
            if action not in recommendations:
                recommendations.append(action)

        risk_tier = (
            "Critical" if prob >= 0.70
            else "High" if prob >= self.threshold
            else "Low"
        )

        return PredictionResponse(
            churn_probability=round(prob, 4),
            decision_threshold=self.threshold,
            risk_tier=risk_tier,
            flag_for_outreach=prob >= self.threshold,
            top_churn_drivers=top_drivers,
            recommended_actions=recommendations,
        )