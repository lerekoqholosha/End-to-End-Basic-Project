# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:46:13 2022
@author: Lereko Qholosha
"""

import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('lr_model.pkl', 'rb') as file:
        data = joblib.load(file)
    return data


data = load_model()

regressor_loaded = data["model"]
hours = data["hours"]


def show_predict_page():
    st.title("Student's Score Prediction")
    
    st.write(""" ### We need some information to predict the marks """)
    
    marks = st.text_input("Enter Number of hours you study: ","")
    
    ok = st.button("Calculate Marks")
    if ok:
        X = np.array([[marks]])
        X = X.astype(float)
        
        marks = regressor_loaded.predict(X)
        st.subheader(f"The estimated Marks are {marks[0]:.2f}") 

st.write('Made 😍 Lereko')
    

