import numpy as np
import matplotlib.pyplot as plt
import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd
import FeatureSelection as fs

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt


def prep_data(data_set):
    if data_set == 1:
        data = pd.read_excel(
            r'C:\\Users\\cbroe\\OneDrive\\Skrivebord\\Stuff\\School\\bachelor\\Python\\Bachelor\\output.xlsx')

        data = data.iloc[13750:86190]

        lon, lat = fs.get_feature_selection_rows(data_set, 10)
        X = data.filter(lon)
        y = data.filter(['lon_rad', 'lat_rad'])

        return data, X, y


    elif data_set == 2:
        data = pd.read_excel(
            r'C:\\Users\\cbroe\\OneDrive\\Skrivebord\\Stuff\\School\\bachelor\\Python\\Bachelor\\output2.xlsx')

        data = data.iloc[0:3700]

        lon, lat = fs.get_feature_selection_rows(data_set, 5)
        X = data.filter(lon)
        y = data.filter(['Long', 'Lat'])

        return data, X, y


# Change this to change dataSet
data_number = 1

data, X, y = prep_data(data_number)

data_length = len(data)
test_train_cut = int(data_length * 0.9)


if data_number == 1:
    lon = data['lon_rad'][:test_train_cut].to_numpy()
    lat = data['lat_rad'][:test_train_cut].to_numpy()
    lon_plot = data['lon_rad'][test_train_cut:].to_numpy()
    lat_plot = data['lat_rad'][test_train_cut:].to_numpy()
    y_lon = data['lon_rad'].values
    y_lat = data['lat_rad'].values

elif data_number == 2:
    lon = data['Long'][:test_train_cut].to_numpy()
    lat = data['Lat'][:test_train_cut].to_numpy()
    lon_plot = data['Long'][test_train_cut:].to_numpy()
    lat_plot = data['Lat'][test_train_cut:].to_numpy()
    y_lon = data['Long'].values
    y_lat = data['Lat'].values


# Splitting the dataset into the Training set and Test set
X_train = X.iloc[:test_train_cut, :]
y_train = y.iloc[:test_train_cut, :]

X_test = X.iloc[test_train_cut:, :]
y_test = y.iloc[test_train_cut:, :]


# Importing the dataset

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

index = np.arange(data_length - test_train_cut)

print(index.shape)
print(lon_plot.shape)
print(y_pred_lon)

plt.subplot(211)
plt.plot(index, lon_plot, 'r', index, y_pred_lon, 'b')
plt.ylabel('Longitude in rad')
plt.xlabel('1 sec interval')
plt.title('Linear Regression')

plt.subplot(212)
plt.plot(index, lat_plot, 'r', index, y_pred_lat, 'b')
plt.ylabel('Latitude in rad')
plt.xlabel('1 sec interval')

plt.show()