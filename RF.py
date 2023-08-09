import pandas as pd
import matplotlib.pyplot as plt
#classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data=pd.read_csv("Iris.csv")
print(data.head())
print(data.isnull().sum())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
