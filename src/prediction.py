import pandas as pd
from src.data_preprocessing import NUMERIC_COLUMNS


def prepare_user_input(
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
):
    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value],
        "education_ Not Graduate": [1 if education == "Not Graduate" else 0],
        "self_employed_ Yes": [1 if self_employed == "Yes" else 0]
    })

    input_data[NUMERIC_COLUMNS] = scaler.transform(
        input_data[NUMERIC_COLUMNS]
    )

    return input_data


def predict_loan_status(model, input_data):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    approval_probability = probability[1]

    return prediction, approval_probability