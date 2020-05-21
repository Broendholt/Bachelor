import numpy as np
import matplotlib.pyplot as plt
import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt

# Importing the dataset
dataset = pd.read_excel(r'C:\\Users\\cbroe\\OneDrive\\Skrivebord\\Stuff\\School\\bachelor\\Python\\Bachelor\\output.xlsx')
X = dataset.iloc[13750:86190, 2:]
X = X.drop(['lon', 'lat', 'lon_rad', 'lat_rad'], axis = 1)
y = dataset.iloc[13750:86190, 10:12]

lon = dataset['lon_rad'][13750:86190:10].to_numpy()
lat = dataset['lat_rad'][13750:86190:10].to_numpy()

index = np.arange(13750, 86190, 10)

# Splitting the dataset into the Training set and Test set
X_train = X.iloc[:65198, :]
y_train = y.iloc[:65198, :]
X_test = X.iloc[65198:, :]
y_test = y.iloc[65198:, :]

# # Data preprocessing for Polynomial Regression
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = SOMETHING)
# X_poly = poly_reg.fit_transform(X)

# Training the Linear Regression model on the Training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred)

# Evaluating the model's performance

MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)
print(MAE)
print(RMSE)
print(R2)

# Plotting
y_pred_lon = y_pred[:,0]
y_pred_lat = y_pred[:,1]
index = np.arange(0, len(lon)-2, 1)


plt.subplot(211)
plt.plot(index, lon[:-2], 'r', index, y_pred_lon, 'b')
plt.ylabel('Longitude in rad')
plt.xlabel('10 sec interval')
plt.title('Linear Regression')

plt.subplot(212)
plt.plot(index, lat[:-2], 'r', index, y_pred_lat, 'b')
plt.ylabel('Latitude in rad')
plt.xlabel('10 sec interval')

plt.show()