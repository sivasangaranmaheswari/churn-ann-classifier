import pandas as pd
# print("Hi")
from keras import Sequential
# print("Hey")
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from Churn_Modelling_Data_Prep import preprocess
import keras

data = pd.read_csv("Churn_Modelling.csv")
X_train, X_test, y_train, y_test = preprocess(data=data)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
print(X_train.dtype)
print(y_train.dtype)
print(X_test.dtype)
print(y_test.dtype)
opt = keras.optimizers.Adam(learning_rate=0.01)

callbacks = [EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=10)]

loss_fn = keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])#, "loss"])

model.fit(X_train, y_train, epochs=100, callbacks=callbacks, validation_data=(X_test, y_test))

model.save("churn_modelling_ann.h5")