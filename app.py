import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import yaml
from yaml.loader import SafeLoader

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL (IMPORTANT FIX) ----------------
model = joblib.load("models/loan_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# ---------------- LOAD USERS ----------------
with open("users.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# ---------------- AUTH ----------------
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login()

# ---------------- SESSION STATE INIT (FIX FOR CHART) ----------------
if "approved" not in st.session_state:
    st.session_state.approved = 0
if "rejected" not in st.session_state:
    st.session_state.rejected = 0

# ---------------- MAIN APP ----------------
if st.session_state["authentication_status"]:

    st.success("Login Successful")
    st.title("🏦 Smart Loan Approval Predictor")

    st.markdown("Check whether a loan will be approved or not.")

    # ---------------- SIDEBAR INPUTS ----------------
    st.sidebar.header("Applicant Information")

    age = st.sidebar.slider("Age", 18, 80, 30)

    gender = st.sidebar.selectbox("Gender", ["male", "female"])

    education = st.sidebar.selectbox(
        "Education",
        ["High School", "Bachelor", "Master", "Doctorate"]
    )

    income = st.sidebar.number_input("Income", min_value=0, value=50000, step=1000)

    home = st.sidebar.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE"]
    )

    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=100000, step=5000)

    if income > 0:
        loan_percent_income = loan_amount / income
    else:
        loan_percent_income = 0

    st.sidebar.write("Loan Percent Income:", round(loan_percent_income, 2))

    intent = st.sidebar.selectbox(
        "Loan Intent",
        ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )

    credit_score = st.sidebar.number_input(
        "Credit Score", min_value=300, max_value=900, value=650, step=1
    )
    previous_loan = st.sidebar.selectbox("Previous Default", ["Yes", "No"])

    loan_percent_income = loan_amount / income if income != 0 else 0

    # ---------------- PREDICTION ----------------
    if st.button("Predict Loan Approval"):

        input_df = pd.DataFrame({
            "person_age": [age],
            "person_gender": [gender],
            "person_education": [education],
            "person_income": [income],
            "person_home_ownership": [home],
            "loan_amnt": [loan_amount],
            "loan_intent": [intent],
            "loan_percent_income": [loan_percent_income],
            "credit_score": [credit_score],
            "previous_loan_defaults_on_file": [previous_loan]
        })

        # Encoding
        for col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

        # Prediction
        result = model.predict(input_df)[0]

        # Update stats
        if result == 1:
            st.success("✅ Loan Approved")
            st.session_state.approved += 1
        else:
            st.error("❌ Loan Not Approved")
            st.session_state.rejected += 1

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Applications", st.session_state.approved + st.session_state.rejected)
    col2.metric("Approved", st.session_state.approved)
    col3.metric("Rejected", st.session_state.rejected)

    # ---------------- PIE CHART ----------------
    pie_data = pd.DataFrame({
        "Status": ["Approved", "Rejected"],
        "Count": [st.session_state.approved, st.session_state.rejected]
    })

    fig = px.pie(
        pie_data,
        values="Count",
        names="Status",
        title="Loan Approval Distribution"
    )

    st.plotly_chart(fig)

# ---------------- AUTH ERROR HANDLING ----------------
elif st.session_state["authentication_status"] is False:
    st.error("Username or Password Incorrect")

elif st.session_state["authentication_status"] is None:
    st.warning("Please Enter Username and Password")
