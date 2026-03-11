from data import fetch_data, load_and_preprocess
from decisiontree import train_and_evaluate_decision_tree, visualise_tree
from svm import train_and_evaluate_svm
from knn import train_and_evaluate_knn, plot_k_vs_accuracy
import pandas as pd

if __name__ == "__main__":
    FILEPATH = "german.data"   # path to your local .data file

    X_train_p, X_test_p, y_train, y_test, preprocessor, feature_names = load_and_preprocess(FILEPATH)

    decision_tree_model = train_and_evaluate_decision_tree(X_train_p, X_test_p, y_train, y_test)
    svm_model           = train_and_evaluate_svm(X_train_p, X_test_p, y_train, y_test)
    knn_model           = train_and_evaluate_knn(X_train_p, X_test_p, y_train, y_test)

    # Optional: visualise k vs accuracy curve
    # plot_k_vs_accuracy(X_train_p, y_train)

    # Optional: visualise the decision tree
    # data  = fetch_data(FILEPATH)
    # df    = pd.DataFrame(data)
    # X_all = df.drop('class', axis=1)
    # y_all = df['class'].map({1: 0, 2: 1})
    # visualise_tree(X_all, y_all, preprocessor, feature_names)