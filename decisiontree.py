from data import fetch_data
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt


def load_and_preprocess(url: str):
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



def train_and_evaluate(X_train_processed, X_test_processed, y_train, y_test):
    """
    Train a DecisionTreeClassifier and print a full evaluation report.

    Returns
    -------
    fitted DecisionTreeClassifier
    """
    model = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)
    model.fit(X_train_processed, y_train)

    y_pred  = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    # ── Metrics ──
    print(f"Accuracy:  {model.score(X_test_processed, y_test):.3f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    print("\nMetrics for 'bad' class (positive = 1):")
    print(f"  Precision : {precision_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"  Recall    : {recall_score(y_test,    y_pred, pos_label=1):.3f}")
    print(f"  F1-score  : {f1_score(y_test,        y_pred, pos_label=1):.3f}")
    print(f"  ROC AUC   : {roc_auc_score(y_test, y_proba):.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['good', 'bad']))

    return model



def visualise_tree(X_all, y_all, preprocessor, feature_names):
    """
    Re-train a shallow tree on the full dataset and plot it.
    """
    X_processed = preprocessor.fit_transform(X_all)

    tree = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=0)
    tree.fit(X_processed, y_all)

    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, class_names=['good', 'bad'],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree – German Credit Data")
    plt.tight_layout()
    plt.show()


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    X_train_p, X_test_p, y_train, y_test, preprocessor, feature_names = load_and_preprocess(URL)

    model = train_and_evaluate(X_train_p, X_test_p, y_train, y_test)

    # Rebuild full X/y for the visualisation (needs original DataFrame)
    data = fetch_data(URL)
    df   = pd.DataFrame(data)
    X_all = df.drop('class', axis=1)
    y_all = df['class'].map({1: 0, 2: 1})
    visualise_tree(X_all, y_all, preprocessor, feature_names)