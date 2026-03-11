import os
import pandas as pd
from data import fetch_data, load_and_preprocess
from decisiontree import train_and_evaluate_decision_tree
from svm import train_and_evaluate_svm
from knn import train_and_evaluate_knn
from visualisations import (plot_original_data,
                             plot_decision_tree_results,
                             plot_decision_tree_structure,
                             plot_svm_results,
                             plot_knn_results)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILEPATH = os.path.join(BASE_DIR, "german.data")

    # ── Load & preprocess ─────────────────────────────────────────────────────
    X_train_p, X_test_p, y_train, y_test, preprocessor, feature_names = load_and_preprocess(FILEPATH)

    # ── 1. Original data visuals ──────────────────────────────────────────────
    raw_df = fetch_data(FILEPATH)
    plot_original_data(raw_df)

    # ── 2. Decision Tree ──────────────────────────────────────────────────────
    decision_tree_model = train_and_evaluate_decision_tree(X_train_p, X_test_p, y_train, y_test)
    plot_decision_tree_results(decision_tree_model, X_test_p, y_test, feature_names)
    plot_decision_tree_structure(decision_tree_model, feature_names)

    # ── 3. SVM ────────────────────────────────────────────────────────────────
    svm_model = train_and_evaluate_svm(X_train_p, X_test_p, y_train, y_test)
    plot_svm_results(svm_model, X_test_p, y_test)

    # ── 4. K-Nearest Neighbours ───────────────────────────────────────────────
    knn_model = train_and_evaluate_knn(X_train_p, X_test_p, y_train, y_test)
    plot_knn_results(knn_model, X_train_p, y_train, X_test_p, y_test)