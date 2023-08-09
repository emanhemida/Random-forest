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
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.89)


model=RandomForestClassifier(n_estimators=20,max_depth=4)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


