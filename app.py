# -*- coding: utf-8 -*-
"""
Created on Fri Feb  25 22:17:46 2022

@author: Bhargav
"""



import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/MLProjects/Red wine/xgboost_wine.pkl','rb'))

def wine_prediction(input_data):

    #changing the input data into numpy array
    input_data_np= np.asarray(input_data)

    #reshape the array as we are predicting for one
    input_reshaped = input_data_np.reshape(1,-1)

    prediction = loaded_model.predict(input_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'This is Low Quality Wine'
        
    else:
        return 'High Quality Wine'

def main():
    #title for web page
    st.title('Red Wine Quality Prediction')
    
    #getting the input data from the user
    
    fixed_acidity = st.number_input('Enter the Fixed Acidity value')
    volatile_acidity = st.number_input('Enter Volatile Acidity value')
    citric_acid = st.number_input('Enter Citric Acid level')
    residual_sugar = st.number_input('Enter residual sugar value')
    chlorides = st.number_input('Enter chlorides  value')
    free_sulfur_dioxide = st.number_input('Enter free sulfur dioxide value')
    total_sulfur_dioxide = st.number_input('Enter total sulfur dioxide value')
    density = st.number_input('Enter density value')
    pH = st.number_input(' Enter pH value')
    sulphates = st.number_input('Enter sulphates value')
    alcohol = st.number_input('Enter Alcohol level')
    
    
    
    #code for prediction
    WineQuality = ''
    
    #creating button
    
    if st.button ('Get Wine Quality'):
        WineQuality = wine_prediction([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
    st.success(WineQuality)
    
    
    
if __name__ == '__main__':
    main()
    