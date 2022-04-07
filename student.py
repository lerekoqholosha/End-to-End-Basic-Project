import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.write('## Student Mark Prediction')

marks=st.text_input("Enter Number of hours you study: ","")

model = joblib.load('lr_model.pkl')
ok = st.button("Calculate Marks")
if ok:
    X=np.array([[marks]])
    X=X.astype(float)

marks = model.predict(X)
st.subheader(f"The estimated Marks are {marks[0]:.2f}")

st.write('Made 😍 Lereko')
    

