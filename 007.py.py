# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 00:20:59 2023

@author: arun
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# load the model from the saved file
with open(r"C:\Users\arun\Music\spider\model (1).sav", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("C:/Users/arun/Music/fff/WineQT.csv")

# create a function to preprocess new data using the same scaler used for training
def preprocess_input(input_data):
    norm = MinMaxScaler()
    normalized_data = norm.fit_transform([input_data])
    return normalized_data

# create the Streamlit app
def app():
    st.title('Wine Quality Prediction App')
    st.write('Enter the wine sample details in the form below to predict its quality.')

    # create the form for input data
    col1, col2 = st.columns(2)
    fixed_acidity = col1.number_input('Fixed Acidity', min_value=0.0, max_value=15.0, value=7.0)
    volatile_acidity = col2.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.5)
    citric_acid = col1.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.5)
    residual_sugar = col2.number_input('Residual Sugar', min_value=0.0, max_value=20.0, value=10.0)
    chlorides = col1.number_input('Chlorides', min_value=0.0, max_value=1.0, value=0.05)
    free_sulfur_dioxide = col2.number_input('Free Sulfur Dioxide', min_value=0, max_value=100, value=50)
    total_sulfur_dioxide = col1.number_input('Total Sulfur Dioxide', min_value=0, max_value=300, value=150)
    density = col2.number_input('Density', min_value=0.900, max_value=1.100, value=1.000)
    pH = col1.number_input('pH', min_value=2.0, max_value=4.0, value=3.0)
    sulphates = col2.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.5)
    alcohol = col1.number_input('Alcohol', min_value=8.0, max_value=16.0, value=10.0)

    # Make predictions with the trained model
    input_data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
    normalized_data = preprocess_input(input_data)
    prediction = model[0].predict(normalized_data)


    # create a button to make the prediction
    if st.button('Predict'):
        # display the prediction result
        if prediction[0] == 0:
            st.write('The model predicts that the new wine sample is of low quality.')
        else:
            st.write('The model predicts that the new wine sample is of good quality.')

# run the app
app()
