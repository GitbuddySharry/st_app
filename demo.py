
import streamlit as st
import numpy as np
st.title("Flower-Type")
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

var= load_iris()
# split the data into x and y
x=var.data
y=var.target
model=KNeighborsClassifier(13)
model.fit(x,y)
#model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
xmin=np.min(x,axis=0)
xmax=np.max(y,axis=0)
sepal_length=st.slider("Sepal Length",xmin[0],xmax[0])
sepal_width=st.slider("Sepal width",xmin[1],xmax[1])
petal_length=st.slider("Petal Length",xmin[2],xmax[2])
petal_width=st.slider("Sepal Width",xmin[3],xmax[3])
y_pred=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

op=["Iris-Setosa","Iris-Versicolor","iris-Virginica"]

st.title(op[y_pred[0]])
