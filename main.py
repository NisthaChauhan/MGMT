'''from data import fetch_data, load_and_preprocess
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt


def train_and_evaluate_svm(X_train_processed, X_test_processed, y_train, y_test):
    """
    Train an RBF-kernel SVC and print a full evaluation report.

    Returns
    -------
    fitted SVC
    """
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    model.fit(X_train_processed, y_train)

    y_pred  = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    print("\n*************SVM EVALUATION*************\n")
    print(f"Accuracy  : {model.score(X_test_processed, y_test):.3f}")

    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    print("\nMetrics for 'bad' class (positive = 1):")
    print(f"  Precision : {precision_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"  Recall    : {recall_score(y_test,    y_pred, pos_label=1):.3f}")
    print(f"  F1-score  : {f1_score(y_test,        y_pred, pos_label=1):.3f}")
    print(f"  ROC AUC   : {roc_auc_score(y_test, y_proba):.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['good', 'bad']))

    return modelfrom data import fetch_data, load_and_preprocess'''
from decisiontree import train_and_evaluate_decision_tree, visualise_tree
from svm import train_and_evaluate_svm
from knn import train_and_evaluate_knn, plot_k_vs_accuracy
import pandas as pd

if __name__ == "__main__":
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    X_train_p, X_test_p, y_train, y_test, preprocessor, feature_names = load_and_preprocess(URL)

    decision_tree_model = train_and_evaluate_decision_tree(X_train_p, X_test_p, y_train, y_test)
    svm_model           = train_and_evaluate_svm(X_train_p, X_test_p, y_train, y_test)
    knn_model           = train_and_evaluate_knn(X_train_p, X_test_p, y_train, y_test)

    # Optional: visualise k vs accuracy curve
    # plot_k_vs_accuracy(X_train_p, y_train)

    # Optional: visualise the decision tree
    # data  = fetch_data(URL)
    # df    = pd.DataFrame(data)
    # X_all = df.drop('class', axis=1)
    # y_all = df['class'].map({1: 0, 2: 1})
    # visualise_tree(X_all, y_all, preprocessor, feature_names)