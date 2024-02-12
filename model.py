import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris=pd.read_csv("iris.data.csv")
X = iris.iloc[:, :-1]  # Select all rows and all columns except the last one as features
y = iris.iloc[:, -1]   # Select all rows and the last column as labels
print(X,y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)

from sklearn.ensemble import RandomForestClassifier
rf_classifier=RandomForestClassifier(max_depth=3,random_state=42)
rf_classifier.fit(x_train,y_train)

import pickle

pickle.dump(rf_classifier,open("model.pkl","wb"))

pickle.dump(scaler, open("scaler.pkl", "wb"))
