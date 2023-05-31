import pandas as pd

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('TTTCFTCCLUWLPEWJACSV')
st.subheader("Step 1: Input a CSV")
uploadedcsv = st.file_uploader(" ")
if uploadedcsv is not None:
    df = pd.read_csv(uploadedcsv)
    st.write(df)

st.subheader("Step 2: Insert Axis Column Titles")
col1,col2,col3 = st.columns(3)
with col1:
    x_column_title = st.text_input('Title of X Value Column')
    d = st.button("Analyze")
with col2:
    y_column_title = st.text_input('Title of Y Value Column')

   
    
#use this to import everything

def calculate_r_squared(x,y):
    correlation_matrix = np.corrcoef(x,y)
    correlation  = correlation_matrix[0,1]
    r_squared = round(correlation**2,4)
    st.subheader(f"{r_squared} is the R^2 for {x_column_title} vs {y_column_title}")

#add column values to list
#Function to create plot and Best Fit
def plot(list1,list2):
    plt.scatter(list1,list2)
    plt.xlabel(x_column_title)
    plt.ylabel(y_column_title)
    coefficients = np.polyfit(x_value_list,y_value_list,1)
    polynomial = np.poly1d(coefficients)
    x_values = np.linspace(min(x_value_list), max(x_value_list),100)
    y_values = polynomial(x_values)
    plt.plot(x_values,y_values, color = 'red')
    plt.title(f'{x_column_title} vs {y_column_title}')
    st.pyplot()
    print("Above is the scatter plot with best fit line.")

st.subheader("Step 3: Click Analyze to Load Scatter Plot with Best Fit Line and Coefficient of Determination")

if d:
    y_value_list = df[y_column_title].tolist() 
    x_value_list = df[x_column_title].tolist()
    plot(x_value_list,y_value_list)
    calculate_r_squared(x_value_list,y_value_list)


