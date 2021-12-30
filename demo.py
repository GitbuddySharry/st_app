
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
xmax=np.max(x,axis=0)
a=xmin[0]
b=xmin[1]
c=xmin[2]
d=xmin[3]
e=xmax[0]
f=xmax[1]
g=xmax[2]
h=xmax[3]
sepal_length=st.slider("Sepal Length",a,e)
sepal_width=st.slider("Sepal width",b,f)
petal_length=st.slider("Petal Length",c,g)
petal_width=st.slider("Sepal Width",d,h)
y_pred=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

op=["Iris-Setosa","Iris-Versicolor","iris-Virginica"]

st.title(op[y_pred[0]])
