import pandas as pd
# print("Hi")
from keras import Sequential
# print("Hey")
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from Churn_Modelling_Data_Prep import preprocess
import keras
import numpy as np
import pickle

model = keras.models.load_model("churn_modelling_ann.h5")
gender_encoder = pickle.load(open("gender_encoder.sav","rb"))
geo_encoder = pickle.load(open("geography_encoder.sav","rb"))
scaler = pickle.load(open("scaler.sav","rb"))

input_data = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 43,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000
}
columns=geo_encoder.get_feature_names_out()
out = geo_encoder.transform([[input_data["Geography"]]]).toarray()[0]
print(out)
input_data.update({col:val for (col, val) in zip(columns, out.tolist())})
del input_data["Geography"]
input_data["Gender"] = gender_encoder.transform([input_data["Gender"]])[0]
print(input_data)
input_data_raw = [list(input_data.values())]

input_data_raw_transform = scaler.transform(input_data_raw)
# print(input_data_raw)
# print(input_data_raw_transform)

predictions = model.predict(input_data_raw_transform)
positive = "Customer is likely to churn"
negative = "Customer is not likely to churn"
print(positive if predictions[0][0] > 0.5 else negative)
