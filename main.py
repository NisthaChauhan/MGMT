from data import fetch_data

url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
data=fetch_data(url)
print(data.head())