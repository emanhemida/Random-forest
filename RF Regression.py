import pandas as pd
#Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data=pd.read_csv("train.csv")
data=data.dropna()


print(data.head())
print(data.isnull().sum())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.89)


model=RandomForestRegressor(n_estimators=200,max_depth=4)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
