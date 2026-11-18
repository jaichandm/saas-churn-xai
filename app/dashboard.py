import streamlit as st
from src.inference import ChurnPredictorService
from src.schema import CustomerInput

st.set_page_config(page_title="SaaS Churn XAI Portal", layout="wide")


@st.cache_resource
def load_service():
    return ChurnPredictorService()


service = load_service()

st.title("SaaS Churn Intelligence & Explainable AI Workspace")
st.markdown(
    f"Cost-calibrated decision threshold: **{service.threshold:.2f}** "
    "(Optimized for $500 FN LTV loss vs. $50 FP retention cost)"
)

with st.sidebar:
    st.header("Account Profile")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=4)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=89.5)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=15000.0, value=float(tenure * monthly_charges)
    )

payload = CustomerInput(
    tenure=tenure,
    Contract=contract,
    InternetService=internet,
    TechSupport=tech_support,
    OnlineSecurity=online_security,
    PaymentMethod=payment,
    MonthlyCharges=monthly_charges,
    TotalCharges=total_charges,
)

result = service.predict(payload)

col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{result.churn_probability:.1%}")
col2.metric("Risk Tier", result.risk_tier)
col3.metric("Outreach Required", "YES" if result.flag_for_outreach else "NO")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Top Risk Drivers (SHAP Impact)")
    if result.top_churn_drivers:
        for d in result.top_churn_drivers:
            st.write(f"- **{d.feature}**: `+{d.shap_impact:.4f}` log-odds contribution")
    else:
        st.success("No positive churn risk factors detected.")

with right:
    st.subheader("Recommended Customer Success Playbook")
    if result.recommended_actions:
        for action in result.recommended_actions:
            st.info(action)
    else:
        st.write("Maintain standard account health cadence.")