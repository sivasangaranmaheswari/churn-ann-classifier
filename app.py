import pandas as pd
# print("Hi")
from keras import Sequential
# print("Hey")
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from Churn_Modelling_Data_Prep import preprocess
import keras
import numpy as np
import streamlit as st
import pickle

st.title("Customer Churn Prediction")

model = keras.models.load_model("churn_modelling_ann.h5", compile=False)
gender_encoder = pickle.load(open("gender_encoder.sav","rb"))
geo_encoder = pickle.load(open("geography_encoder.sav","rb"))
scaler = pickle.load(open("scaler.sav","rb"))

geography = st.selectbox("Geography", geo_encoder.categories_[0])
gender = st.selectbox("Gender", gender_encoder.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
creit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
number_of_products = st.slider("No. of products", 0, 5)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

input_data = {
    "CreditScore": creit_score,
    "Gender": gender_encoder.transform([gender])[0],
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": number_of_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}

# st.text()
columns=geo_encoder.get_feature_names_out()
values = geo_encoder.transform([[geography]]).toarray()[0].tolist()
input_data.update({col:val for col, val in zip(columns, values)})

scaled_input_data = scaler.transform([list(input_data.values())])
predictions = model.predict(scaled_input_data)

positive = "Customer is likely to churn"
negative = "Customer is not likely to churn"

st.text(positive if predictions[0][0] > 0.5 else negative)

