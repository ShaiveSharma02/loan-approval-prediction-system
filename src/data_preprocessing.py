import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


NUMERIC_COLUMNS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    return data


def preprocess_data(data):
    data = data.copy()

    data = data.ffill()

    data["loan_status"] = data["loan_status"].str.strip()
    data["target"] = data["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })

    data = pd.get_dummies(
        data,
        columns=["education", "self_employed"],
        drop_first=True
    )

    X = data.drop(["loan_id", "loan_status", "target"], axis=1)
    y = data["target"]

    scaler = MinMaxScaler()
    X[NUMERIC_COLUMNS] = scaler.fit_transform(X[NUMERIC_COLUMNS])

    return X, y, scaler


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )