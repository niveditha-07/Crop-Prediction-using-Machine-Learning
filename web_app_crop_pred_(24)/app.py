
import streamlit as st
st.header('Crop Prediction System')

import pandas as pd
dataset=pd.read_csv('crop.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
y=pd.get_dummies(y)
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#print(x_train.shape)
#print(x_test.shape)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=4)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
#print(y_pred)
crops=list(y.columns)
from sklearn.metrics import accuracy_score
#print(accuracy_score(y_pred,y_test))
N=st.text_input('Enter Nitrogen')
P=st.text_input('Enter Phosporous')
K=st.text_input('Enter Potassium')
T=st.text_input('Enter Temperature')
H=st.text_input('Enter Humidity')
PH=st.text_input('Enter PH')
R=st.text_input('Enter Rainfall')

if st.button('Predit Crop'):
    crop=classifier.predict([[N,P,K,T,H,PH,R]])[0]
    crop=list(crop)
    i=crop.index(1)
    st.success(crops[i])
