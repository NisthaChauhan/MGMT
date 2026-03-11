from data import fetch_data, load_and_preprocess
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt


def find_best_k(X_train_processed, y_train, k_range=range(1, 21)):
    """
    Use cross-validation to find the optimal value of k.

    Returns
    -------
    best_k : int
    """
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        scores = cross_val_score(knn, X_train_processed, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    best_k = k_range[np.argmax(cv_scores)]
    print(f"Best k found via 5-fold CV: {best_k} (accuracy = {max(cv_scores):.3f})")
    return best_k


def train_and_evaluate_knn(X_train_processed, X_test_processed, y_train, y_test,
                           n_neighbors=None):
    """
    Train a KNeighborsClassifier and print a full evaluation report.
    If n_neighbors is None, the best k is chosen via cross-validation.

    Returns
    -------
    fitted KNeighborsClassifier
    """
    if n_neighbors is None:
        n_neighbors = find_best_k(X_train_processed, y_train)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p=2)
    model.fit(X_train_processed, y_train)

    y_pred  = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    print("\n*************K-NEAREST NEIGHBOURS EVALUATION*************\n")
    print(f"n_neighbors : {n_neighbors}")
    print(f"Accuracy    : {model.score(X_test_processed, y_test):.3f}")

    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    print("\nMetrics for 'bad' class (positive = 1):")
    print(f"  Precision : {precision_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"  Recall    : {recall_score(y_test,    y_pred, pos_label=1):.3f}")
    print(f"  F1-score  : {f1_score(y_test,        y_pred, pos_label=1):.3f}")
    print(f"  ROC AUC   : {roc_auc_score(y_test, y_proba):.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['good', 'bad']))

    return model


def plot_k_vs_accuracy(X_train_processed, y_train, k_range=range(1, 21)):
    """
    Plot cross-validated accuracy against k and display the chart.
    """
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        scores = cross_val_score(knn, X_train_processed, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), cv_scores, marker='o', linewidth=2, color='steelblue')
    plt.axvline(x=list(k_range)[np.argmax(cv_scores)], color='red',
                linestyle='--', label=f"Best k = {list(k_range)[np.argmax(cv_scores)]}")
    plt.xlabel("Number of Neighbours (k)")
    plt.ylabel("5-Fold CV Accuracy")
    plt.title("KNN – Accuracy vs k (German Credit Data)")
    plt.legend()
    plt.tight_layout()
    plt.show()