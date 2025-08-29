import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
data = pd.read_csv("Churn_Modelling.csv")

def preprocess(data, is_main=False):
    data = data[data.columns[3:]]
    data = encode_categorical_features(data=data)
    features, targets = data.drop(["Exited"], axis=1), data["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=101)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pickle.dump(scaler, open("scaler.sav", "wb"))
    if is_main:
        print("Generated Training Data")
    return X_train, X_test, y_train, y_test

def encode_categorical_features(data):
    geo_encoder = OneHotEncoder()
    out = geo_encoder.fit_transform(data[["Geography"]])
    geo_encoded_df = pd.DataFrame(out.toarray(), columns=geo_encoder.get_feature_names_out())
    data = pd.concat((data.drop("Geography", axis=1), geo_encoded_df), axis=1)
    pickle.dump(geo_encoder, open("geography_encoder.sav","wb"))
    label_encoder = LabelEncoder()
    data.Gender = label_encoder.fit_transform(data.Gender)
    pickle.dump(label_encoder, open("gender_encoder.sav","wb"))
    return data

if __name__ == "__main__":
    preprocess(data=data, is_main=True)