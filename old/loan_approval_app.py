import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Random Forest model and scaler
rf_model = RandomForestClassifier()
rf_model_under = RandomForestClassifier()
scaler = MinMaxScaler()
# Load the dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Handle missing values
data.fillna(method="ffill", inplace=True)
data.columns = data.columns.str.strip()

# Encode categorical features
data = pd.get_dummies(data, columns=["education", "self_employed"], drop_first=True)

# Feature scaling
numeric_columns = ["no_of_dependents", "income_annum", "loan_amount", "loan_term", "cibil_score",
                   "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Split features and target
X = data.drop(["loan_status", "loan_id"], axis=1)  # Exclude loan_id column
y = data["loan_status"]
y_binary = (y == ' Approved').astype(int)

# Handle class imbalance using SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y_binary)

# Handle class imbalance using under-sampling
under_sampler = RandomUnderSampler()
X_resampled_under, y_resampled_under = under_sampler.fit_resample(X_resampled, y_resampled)

# Build and train a Random Forest model with under-sampled data
rf_model_under.fit(X_resampled_under, y_resampled_under)

# Streamlit App
st.title("Loan Approval Prediction")

st.sidebar.header("User Input")

st.set_option('deprecation.showPyplotGlobalUse', False)

# Create input fields for user to provide feature values
no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
income_annum = st.sidebar.number_input("Annual Income", min_value=0, value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=0)
loan_term = st.sidebar.number_input("Loan Term (Months)", min_value=0, value=0)
cibil_score = st.sidebar.number_input("CIBIL Score", min_value=300, max_value=900, value=300)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, value=0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, value=0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0, value=0)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, value=0)
education_graduate = st.sidebar.checkbox("Graduate")
self_employed_yes = st.sidebar.checkbox("Self Employed")

# Create a DataFrame from user inputs
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
    "education_ Not Graduate": [1 if not education_graduate else 0],  # Adjust categorical encoding
    "self_employed_ Yes": [1 if self_employed_yes else 0]  # Adjust categorical encoding
})

# Preprocess user input using the same scaler
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

# Predict using the under-sampled model
prediction = rf_model_under.predict(input_data)

st.header("Prediction Result")

if prediction[0] == 1:
    st.success("Congratulations! The loan is likely to be approved.")
else:
    st.warning("Sorry, the loan is likely to be rejected.")

# Visualize confusion matrix for under-sampled model
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_resampled_under, rf_model_under.predict(X_resampled_under)), annot=True, cmap='Greens', fmt='g')
st.pyplot()

# Visualize ROC Curve for under-sampled model
plt.figure(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_resampled_under, rf_model_under.predict_proba(X_resampled_under)[:, 1])
plt.plot(fpr, tpr, label="Random Forest (Under-sampled)")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
st.pyplot()

# Visualize Precision-Recall Curve for under-sampled model
plt.figure(figsize=(10, 6))
precision, recall, _ = precision_recall_curve(y_resampled_under, rf_model_under.predict_proba(X_resampled_under)[:, 1])
plt.plot(recall, precision, label="Random Forest (Under-sampled)")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
st.pyplot()






