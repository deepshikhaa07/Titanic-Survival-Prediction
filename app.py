import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('logistic_model_titanic.pkl')

# Title
st.title("🚢 Titanic Survival Prediction App")

# Sidebar inputs
st.sidebar.header("Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 50.0)

# Map sex to numeric
sex_num = 1 if sex == "male" else 0

# Create input dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_num],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have SURVIVED.")
    else:
        st.error("💀 The passenger would NOT have survived.")
