from data import fetch_data, load_and_preprocess
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt




def train_and_evaluate_decision_tree(X_train_processed, X_test_processed, y_train, y_test):
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
    print("*************DECISION TREE EVALUATION*************\N")
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


# ── 3. FULL-DATA TREE VISUALISATION ─────────────────────────────────────────

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