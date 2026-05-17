from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_resampled, y_resampled)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )
    matrix = confusion_matrix(y_test, predictions)

    return accuracy, report, matrix