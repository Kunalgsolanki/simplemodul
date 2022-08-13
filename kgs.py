
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
st.set_page_config(page_title='kgs modual')
st.title("Kgs modual")

st.write("""

# Salary predication Modal
Salary vs.  *Experience*

"""
)
# read data 
df = pd.read_csv('Salary_Data.csv')
# print(df)
# Take the datapoint  spit lebal and feature
# st.write(df)
# here salary is depedend veriable and experince indipended veriable
x =df.iloc[:,[0]] .values
y=df.iloc[:,[-1]] .values

x_train,x_test,y_train,y_test   =   train_test_split(x,y, test_size=0.3 ,random_state=0 )



exp = st.sidebar.slider("Exprience",1,10,2)




reg = LinearRegression()
reg.fit(x_train,y_train)





y_pred = reg.predict([[exp]])

st.write(f"Experience",exp)

st.write  (f"Salaray:",float(y_pred))


st.write("""

# Scatter plot

Salary vs.  *Experience*

"""
)

kgs = plt.figure()
plt.scatter(x,y,alpha=0.9,cmap='viridis')

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.colorbar()
st.pyplot(kgs)
# python -m streamlit run kgs.py