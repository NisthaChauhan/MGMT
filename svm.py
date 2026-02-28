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


def train_and_evaluate_svm(X_train_processed, X_test_processed, y_train, y_test):
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("ROC AUC Score:", roc_auc_score(y_test, model.decision_function(X_test_processed)))
