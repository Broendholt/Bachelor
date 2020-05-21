import numpy as np
import matplotlib.pyplot as plt
import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd

from sklearn.metrics import r2_score

import ConfigFile as cf
from os import listdir
from os.path import join, splitext

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt


data_path = cf.get_user_data()
data_path = join(data_path, 'output.xlsx')

data_excel = pd.read_excel(data_path)

length = len(data_excel)

fromIndex = 18000
toIndex = length
steps = 1

lon = data_excel['lon_rad'][fromIndex:toIndex:steps].to_numpy()
lat = data_excel['lat_rad'][fromIndex:toIndex:steps].to_numpy()
date_time = data_excel['datetime'][fromIndex:toIndex:steps].to_numpy()

lon_min = 2
lon_max = 0
lon_avg = 0
lon_count_only_values = 0


def show_data_in_plot(data):

    data_min = 2
    data_max = 0
    data_avg = 0
    data_count_only_values = 0

    for i in range(len(data)):
        if data[i] < data_min and data[i] != 0:
            data_min = data[i]

        if data[i] > data_max and data[i] != 0:
            data_max = data[i]

        if data[i] != 0 and not np.isnan(data[i]):
            data_avg += data[i]
            data_count_only_values += 1

    data_avg /= data_count_only_values

    print(data_min, " : ", data_max, " : ", data_avg)

    plt.hlines(y=data_max, xmin=0, xmax=len(data))
    plt.hlines(y=data_min, xmin=0, xmax=len(data))
    plt.hlines(y=data_avg, xmin=0, xmax=len(data))

    plt.plot(data)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()


lon = data_excel['lon_rad'][13750:86190:10].to_numpy()
lat = data_excel['lat_rad'][13750:86190:10].to_numpy()

index = np.arange(13750, 86190, 10)




dataset = pd.read_excel(r'C:\\Users\\cbroe\\OneDrive\\Skrivebord\\Stuff\\School\\bachelor\\Python\\Bachelor\\output.xlsx')
X = dataset.iloc[13750:86190, 2:]
X = X.drop(['lon', 'lat', 'lon_rad', 'lat_rad'], axis = 1)
y = dataset.iloc[13750:86190, 10:12]

X_train = X.iloc[:65198, :]
y_train = y.iloc[:65198, :]
X_test = X.iloc[65198:, :]
y_test = y.iloc[65198:, :]

# Feature Scaling
from sklearn.preprocessing import StandardScaler    # we should get values <-3, 3>

sc_X_train = StandardScaler()
sc_X_test = StandardScaler()
sc_y_train = StandardScaler()
sc_y_test = StandardScaler()

X_train = sc_X_train.fit_transform(X_train)
X_test = sc_X_test.fit_transform(X_test)
y_train = sc_y_train.fit_transform(y_train)
y_test = sc_y_test.fit_transform(y_test)

# Defining the model
model = Sequential()
model.add(Dense(30, input_shape=(19,), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2,))
model.compile(Adam(lr=0.003), 'mean_squared_error')

# Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'
#earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

# Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history
history = model.fit(X_train, y_train, epochs = 45, validation_split = 0.2, verbose = 1)

# Plots 'history'
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(loss_values, 'bo', label='training loss')
plt.plot(val_loss_values, 'r', label='training loss val')

# Runs model with its current weights on the training and testing data
y_train_pred = sc_y_train.inverse_transform(model.predict(X_train))
y_test_pred = sc_y_test.inverse_transform(model.predict(X_test))
y_train = sc_y_train.inverse_transform(y_train)
y_test = sc_y_test.inverse_transform(y_test)
print(y_test_pred)

"""
# Calculates and prints r2 score of training and testing data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
MAE_test = mean_absolute_error(y_test, y_test_pred)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
MAE_train = mean_absolute_error(y_train, y_train_pred)
RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(MAE_train)
print(RMSE_train)
print(MAE_test)
print(RMSE_test)
"""


y_test_pred_lon = y_test_pred[:,0]
y_test_pred_lat = y_test_pred[:,1]

print(lon)
print(lat)
print(y_test_pred_lon)
print(y_test_pred_lat)

index = np.arange(0, len(lon)-2, 1)

"""
plt.plot(lon[:-2], 'r-', lat[:-2], 'b-',
         y_test_pred_lon, 'r--',
         y_test_pred_lat, 'b--')
"""

plt.subplot(211)
plt.plot(index, lon[:-2], 'r', index, y_test_pred_lon, 'b')
plt.ylabel('Longitude in rad')
plt.xlabel('10 sec interval')
plt.title('Multilayer Perception')

plt.subplot(212)
plt.plot(index, lat[:-2], 'r', index, y_test_pred_lat, 'b')
plt.ylabel('Latitude in rad')
plt.xlabel('10 sec interval')

plt.show()


