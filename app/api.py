from fastapi import FastAPI
from src.inference import ChurnPredictorService
from src.schema import CustomerInput, PredictionResponse

app = FastAPI(
    title="SaaS Churn Intelligence & XAI API",
    description="Cost-calibrated churn scoring with real-time SHAP attribution.",
    version="1.0.0",
)

predictor = ChurnPredictorService()


@app.get("/health")
def health_check():
    return {"status": "healthy", "optimal_threshold": predictor.threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict_customer_churn(payload: CustomerInput):
    return predictor.predict(payload)