from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
def attributes():
	cols = ["week", "center_id", "meal_id", "base_price", "emailer_for_promotion", "homepage_featured", "num_orders"]
	df = pd.read_csv( "ibm1.csv")
	return df
def preprocess(df, train, test):
	continuous = ["week", "center_id", "meal_id", "base_price", "emailer_for_promotion", "homepage_featured"]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])
	trainX = np.hstack([trainContinuous])
	testX = np.hstack([testContinuous])
	return (trainX, testX)
df = attributes()
(train, test) = train_test_split(df, test_size=0.20)
maxPrice = train["num_orders"].max()
trainY = train["num_orders"] 
testY = test["num_orders"]
(trainX, testX) = preprocess(df, train, test)
dim=trainX.shape[1]
model = Sequential()
model.add(Dense(1500, input_dim=dim, activation="relu"))#1
model.add(Dense(1500, activation="relu"))#2
model.add(Dense(1500, activation="relu"))#3
model.add(Dense(1500, activation="relu"))#4
model.add(Dense(1500, activation="relu"))#5
model.add(Dense(1500, activation="relu"))#6
model.add(Dense(1500, activation="relu"))#7
model.add(Dense(1500, activation="relu"))#8
model.add(Dense(1500, activation="relu"))#9
model.add(Dense(1500, activation="relu"))#10
model.add(Dense(1500, activation="relu"))#11
model.add(Dense(1500, activation="relu"))#12
model.add(Dense(1, activation="linear"))#13
#opt= Adam(lr=0.001)
model.compile(loss='mse', optimizer='adam')
model.fit(x=trainX, y=trainY, validation_data=(testX, testY),epochs=10000, batch_size=10240)
df_test = pd.read_csv('test.csv')
columns_to_train = ["week", "center_id", "meal_id", "base_price", "emailer_for_promotion", "homepage_featured"]
X = df_test[columns_to_train]
preds = model.predict(X)
prediction_df = df_test.copy()
prediction_df['num_orders'] = preds
prediction_df.to_csv('prediction4.csv', index=False)
