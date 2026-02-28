from data import fetch_data
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def load_and_preprocess_svm(url: str):
    """
    Fetch raw data, build a DataFrame, encode the target, and return
    train/test splits together with the fitted preprocessor and feature names.

    Returns
    -------
    X_train_processed, X_test_processed, y_train, y_test,
    preprocessor, feature_names
    """
    # Load
    data = fetch_data(url)
    df = pd.DataFrame(data)
    print(df.head())

    # Separate features / target
    X = df.drop('class', axis=1)
    y = df['class'].map({1: 0, 2: 1})   # 0 = good, 1 = bad

    categorical_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings',
        'employment_since', 'personal_status', 'other_debtors',
        'property', 'other_installment_plans', 'housing', 'job',
        'telephone', 'foreign_worker'
    ]
    numerical_cols = [
        'duration', 'credit_amount', 'installment_rate',
        'residence_since', 'age', 'existing_credits', 'dependents'
    ]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Fit preprocessor on train, transform both splits
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                               unknown_value=-1), categorical_cols)
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    feature_names = numerical_cols + categorical_cols

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


def train_and_evaluate_svm(X_train_processed, X_test_processed, y_train, y_test):
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("ROC AUC Score:", roc_auc_score(y_test, model.decision_function(X_test_processed)))
