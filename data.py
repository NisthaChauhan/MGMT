import pandas as pd
import requests
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

'''
# URL of the original dataset (categorical attributes)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
'''
# Column names as described in the documentation
columns = [
    'checking_status',      # Attribute 1
    'duration',             # Attribute 2
    'credit_history',       # Attribute 3
    'purpose',              # Attribute 4
    'credit_amount',        # Attribute 5
    'savings',              # Attribute 6
    'employment_since',     # Attribute 7
    'installment_rate',     # Attribute 8
    'personal_status',      # Attribute 9
    'other_debtors',        # Attribute 10
    'residence_since',      # Attribute 11
    'property',             # Attribute 12
    'age',                  # Attribute 13
    'other_installment_plans', # Attribute 14
    'housing',              # Attribute 15
    'existing_credits',     # Attribute 16
    'job',                  # Attribute 17
    'dependents',           # Attribute 18
    'telephone',            # Attribute 19
    'foreign_worker',       # Attribute 20
    'class'                 # Target: 1 = good, 2 = bad
]

def fetch_data(url):
        
    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()  
        # Load into pandas DataFrame (space-separated, no header)
        data = pd.read_csv(StringIO(response.text), sep=' ', header=None, names=columns)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        print("Please download the file manually from:")
        print(url)
        print("and then load it with pd.read_csv('german.data', sep=' ', header=None, names=columns)")

    return data

'''if __name__ == "__main__":
    data = fetch_data(url)
'''


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

