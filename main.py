'''from data import fetch_data
import pandas as pd
url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
data=fetch_data(url)
print(data.head())

df=pd.DataFrame(data)
cols=df.columns
for i in cols:
    print(df[i].value_counts(),"\n*********************\n")
'''
from data import fetch_data
from decisiontree import load_and_preprocess, train_and_evaluate, visualise_tree
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