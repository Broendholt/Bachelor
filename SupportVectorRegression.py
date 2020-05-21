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
X = X.values

y_lon = dataset.iloc[13750:86190, 10].values
y_lat = dataset.iloc[13750:86190, 11].values

lon = dataset['lon_rad'][13750:86190:10].to_numpy()
lat = dataset['lat_rad'][13750:86190:10].to_numpy()

index = np.arange(13750, 86190, 10)


# Feature Scaling
from sklearn.preprocessing import StandardScaler    # we should get values <-3, 3>

sc_X = StandardScaler()
sc_y_lon = StandardScaler()
sc_y_lat = StandardScaler()

X = sc_X.fit_transform(X)

y_lon = y_lon.reshape(-1, 1)
y_lon = sc_y_lon.fit_transform(y_lon)

y_lat = y_lat.reshape(-1, 1)
y_lat = sc_y_lat.fit_transform(y_lat)

print(X)
print(y_lon)
print(y_lat)

# Splitting the dataset into the Training set and Test set
X = pd.DataFrame(X)
y_lon = pd.DataFrame(y_lon)
y_lat = pd.DataFrame(y_lat)

X_train = X.iloc[:65198, :]
y_lon_train = y_lon.iloc[:65198, :]
y_lat_train = y_lat.iloc[:65198, :]

X_test = X.iloc[65198:, :]
y_lon_test = y_lon.iloc[65198:, :]
y_lat_test = y_lat.iloc[65198:, :]

# Training the SVR model for longitude
from sklearn.svm import SVR
regressor_lon = SVR(kernel='rbf')
regressor_lon.fit(X_train, y_lon_train)

# Training the SVR model for latitude
regressor_lat = SVR(kernel='rbf')
regressor_lat.fit(X_train, y_lat_train)

# Predicting a new result for longitude
y_lon_pred = sc_y_lon.inverse_transform(regressor_lon.predict(X_test))   # rescaled predictions
y_lon_pred = pd.DataFrame(y_lon_pred)

# Predicting a new result for latitude
y_lat_pred = sc_y_lat.inverse_transform(regressor_lat.predict(X_test))   # rescaled predictions
y_lat_pred = pd.DataFrame(y_lat_pred)

# Evaluating the model's performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAE_lon = mean_absolute_error(y_lon_test, y_lon_pred)
MAE_lat = mean_absolute_error(y_lat_test, y_lat_pred)

RMSE_lon = np.sqrt(mean_squared_error(y_lon_test, y_lon_pred))
RMSE_lat = np.sqrt(mean_squared_error(y_lat_test, y_lat_pred))

R2_lon = r2_score(y_lon_test, y_lon_pred)
R2_lat = r2_score(y_lat_test, y_lat_pred)

print(MAE_lon)
print(RMSE_lon)
print(R2_lon)
print(MAE_lat)
print(RMSE_lat)
print(R2_lat)

# Plotting
y_pred_lon = y_lon_pred
y_pred_lat = y_lat_pred
index = np.arange(0, len(lon)-2, 1)


plt.subplot(211)
plt.plot(index, lon[:-2], 'r', index, y_pred_lon, 'b')
plt.ylabel('Longitude in rad')
plt.xlabel('10 sec interval')
plt.title('Support Vector Regression')

plt.subplot(212)
plt.plot(index, lat[:-2], 'r', index, y_pred_lat, 'b')
plt.ylabel('Latitude in rad')
plt.xlabel('10 sec interval')

plt.show()