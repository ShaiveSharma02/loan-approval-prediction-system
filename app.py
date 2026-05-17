import streamlit as st

from src.data_preprocessing import (
    load_data,
    preprocess_data,
    split_data
)
from src.model_training import train_model, evaluate_model
from src.prediction import prepare_user_input, predict_loan_status
from src.visualization import plot_confusion_matrix


DATA_PATH = "data/loan_approval_dataset.csv"


@st.cache_resource
def build_model():
    data = load_data(DATA_PATH)

    X, y, scaler = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    accuracy, report, matrix = evaluate_model(
        model,
        X_test,
        y_test
    )

    return model, scaler, accuracy, report, matrix


st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Prediction System")

st.write(
    "A machine learning web application that predicts whether a loan application "
    "is likely to be approved or rejected based on applicant financial details."
)

model, scaler, accuracy, report, matrix = build_model()

st.sidebar.header("Applicant Details")

no_of_dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=2
)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

income_annum = st.sidebar.number_input(
    "Annual Income",
    min_value=0,
    value=5000000,
    step=100000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    value=15000000,
    step=100000
)

loan_term = st.sidebar.number_input(
    "Loan Term",
    min_value=1,
    max_value=30,
    value=10
)

cibil_score = st.sidebar.slider(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

residential_assets_value = st.sidebar.number_input(
    "Residential Assets Value",
    min_value=0,
    value=5000000,
    step=100000
)

commercial_assets_value = st.sidebar.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=2000000,
    step=100000
)

luxury_assets_value = st.sidebar.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=3000000,
    step=100000
)

bank_asset_value = st.sidebar.number_input(
    "Bank Asset Value",
    min_value=0,
    value=4000000,
    step=100000
)

if st.sidebar.button("Predict Loan Status"):

    input_data = prepare_user_input(
        scaler,
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    )

    prediction, approval_probability = predict_loan_status(
        model,
        input_data
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Loan is likely to be Approved")
    else:
        st.error("Loan is likely to be Rejected")

    st.metric(
        "Approval Probability",
        f"{approval_probability * 100:.2f}%"
    )

st.divider()

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

with col2:
    st.write("Model Used: Random Forest Classifier")

st.subheader("Confusion Matrix")

fig = plot_confusion_matrix(matrix)
st.pyplot(fig)

st.subheader("Classification Report")

st.dataframe(report)