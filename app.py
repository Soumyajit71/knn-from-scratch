import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from KNeighboursClassifier import Knn


df=pd.read_csv('Social_Network_Ads.csv')
df=df.iloc[:,1:]
encoder=LabelEncoder()
df['Gender']=encoder.fit_transform(df['Gender'])
scaler=StandardScaler()
X=df.iloc[:,0:-1].values
X=scaler.fit_transform(X)
y=df.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("Using Built in sklearn model: ",accuracy_score(y_test,y_pred))

#print(X.head())

"""newknn=knn(k=5)
newknn.fit(X_train,y_train)
newknn.predict(X_test)"""
apnaknn=Knn(k=5)
apnaknn.fit(X_train,y_train)
y_pred1=apnaknn.predict(X_test)
print("Using Our user defined KNN Class: ",accuracy_score(y_test,y_pred1))