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

# Importing the dataset
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

elif data_number == 2:
    lon = data['Long'][:test_train_cut].to_numpy()
    lat = data['Lat'][:test_train_cut].to_numpy()
    lon_plot = data['Long'][test_train_cut:].to_numpy()
    lat_plot = data['Lat'][test_train_cut:].to_numpy()


# Splitting the dataset into the Training set and Test set
X_train = X.iloc[:test_train_cut, :]
y_train = y.iloc[:test_train_cut, :]

X_test = X.iloc[test_train_cut:, :]
y_test = y.iloc[test_train_cut:, :]


# Training the Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50)  # here I would build one tree for each transit
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred)

# Evaluating the model's performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)
print(MAE)
print(RMSE)
print(R2)

# Plotting
y_pred_lon = y_pred[:,0]
y_pred_lat = y_pred[:,1]
index = np.arange(data_length - test_train_cut)

plt.subplot(211)
plt.plot(index, lon_plot, 'r', index, y_pred_lon, 'b')
plt.ylabel('Longitude in rad')
plt.xlabel('1 sec interval')
plt.title('Random Forrest')

plt.subplot(212)
plt.plot(index, lat_plot, 'r', index, y_pred_lat, 'b')
plt.ylabel('Latitude in rad')
plt.xlabel('1 sec interval')

plt.show()