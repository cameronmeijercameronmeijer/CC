import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
from functools import cache
import numpy as np

st.header("Machine Learning for Survey Data")
# Load the survey response data into a pandas DataFrame
@cache
def getdata():
    data = pd.read_csv('Data1010.csv')
    return data
data = getdata()


# Split the data into input features (X) and target variable (y)
X = data[['DOTW', 'Checkout Experience', 'Team Member Rating']]  # Select relevant features
y = data['Overall Satisfaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# Assuming you have trained and saved your machine learning model as 'model'

# Take inputs from the user
input_feature1 = st.slider("Input Day of The Week. 1 = Sunday, 2 = Monday, 3 = Tuesday...",1,7,1,1)
input_feature2 = st.slider('Input Checkout Experience Rating: ',1,10,1,1)
input_feature3 = st.slider('Team Member Friendliness Rating: ',1,10,1,1)
# input_feature1 = float(input("Enter feature 1: "))
# input_feature2 = float(input("Enter feature 2: "))
# input_feature3 = float(input("Enter feature 3: "))

# Create a dictionary with the input values
button=st.button("Calculate")

if button:
    input_data = {
        'DOTW': [input_feature1],
        'Checkout Experience': [input_feature2],
        'Team Member Rating': [input_feature3]
    }

# Create a pandas DataFrame from the input data
    input_df = pd.DataFrame(input_data)

# Make predictions using the trained model
    predictions = model.predict(input_df)

# Display the predicted value
    prediction = round(predictions[0],2)
    st.write("Predicted value:", str(prediction))
