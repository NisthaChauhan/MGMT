import pandas as pd
import requests
from io import StringIO

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
        response.raise_for_status()  # Check for download errors

        # Load into pandas DataFrame (space-separated, no header)
        data = pd.read_csv(StringIO(response.text), sep=' ', header=None, names=columns)

        # Display the first 5 rows
        print("First 5 rows of the German Credit dataset:")
        print(data.head())

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        print("Please download the file manually from:")
        print(url)
        print("and then load it with pd.read_csv('german.data', sep=' ', header=None, names=columns)")

    return data

if __name__ == "__main__":
    data = fetch_data(url)
