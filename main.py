from data import fetch_data
import pandas as pd
url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
data=fetch_data(url)
print(data.head())

df=pd.DataFrame(data)
cols=df.columns
'''for i in cols:
    print(df[i].value_counts(),"\n*********************\n")'''

