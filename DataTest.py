import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


existing_folders = []

def check_if_folders_exist():
    data_path = cf.get_user_data()
    files_in_folder = listdir(data_path)

    i = 0

    for folder in files_in_folder:
        if ".zip" not in folder:

            if cf.data_structure.get(folder):
                print("==============================================")
                print("Folder name:      " + folder)
                print("Folder info:")
                print("Data type:        " + cf.data_structure[folder]["data_type"])
                print("Data extension:   " + cf.data_structure[folder]["data_extension"])
                print("Data Description: " + cf.data_structure[folder]["data_description"])
                print("==============================================\n")

                existing_folders.append(folder)
                i += 1
    if i is 3:
        print("Found all folders 3/3")
        print("All data is found")
    else:
        print("Found only ", i," folders out of 3")


def check_if_data_can_be_read():
    data_path = cf.get_user_data()

    for folder in existing_folders:
        files = listdir(join(data_path, folder))

        for file in files:
            file_ext = splitext(file)[1]

            full_file_path = join(data_path, join(folder, file))

            can_read_excel = False
            can_read_pickle = False
            can_read_csv = False

            if file_ext.replace('.', '') == cf.data_structure[folder]["data_extension"]:
                if cf.data_structure[folder]["data_extension"] == "xlsx":
                    read_excel_file(full_file_path)
                    break

                elif cf.data_structure[folder]["data_extension"] == "pkl":
                    read_pickle_file(full_file_path)
                    break

                elif cf.data_structure[folder]["data_extension"] == "csv":
                    read_csv_file(full_file_path)
                    break

            if can_read_excel:
                print("Excel files ")


def read_excel_file(file_path):
    try:
        data = pd.read_excel(file_path, index_col=0)
        print("can read excel file")
    except:
        print("can't read excel file")


def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=";", encoding='utf-8')
        print("can read csv file")
    except:
        print("can't read csv file")

def read_pickle_file(file_path):
    try:
        data = pd.read_pickle(file_path)
        print("can read pickle file")
    except:
        print("can't read pickle file")


check_if_folders_exist()
check_if_data_can_be_read()



# ALGORITHMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# LINEAR REGRESSION


# # Importing the dataset
# dataset = pd.read_excel(r'C:\Users\mstep\Desktop\Bachelor Project\Data\output.xlsx')
# X = dataset.iloc[13750:86190, 2:]
# X = X.drop(['lon', 'lat', 'lon_rad', 'lat_rad'], axis = 1)
# y = dataset.iloc[13750:86190, 10:12]
#
# # Splitting the dataset into the Training set and Test set
# X_train = X.iloc[:65198, :]
# y_train = y.iloc[:65198, :]
# X_test = X.iloc[65198:, :]
# y_test = y.iloc[65198:, :]
#
# # # Data preprocessing for Polynomial Regression
# # from sklearn.preprocessing import PolynomialFeatures
# # poly_reg = PolynomialFeatures(degree = SOMETHING)
# # X_poly = poly_reg.fit_transform(X)
#
# # Training the Linear Regression model on the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# y_pred = pd.DataFrame(y_pred)
#
# # Evaluating the model's performance
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# MAE = mean_absolute_error(y_test, y_pred)
# RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
# R2 = r2_score(y_test, y_pred)
# print(MAE)
# print(RMSE)
# print(R2)



# SUPPORT VECTOR REGRESSION (SVR)


# Importing the dataset
dataset = pd.read_excel(r'C:\Users\mstep\Desktop\Bachelor Project\Data\output.xlsx')
X = dataset.iloc[13750:86190, 2:]
X = X.drop(['lon', 'lat', 'lon_rad', 'lat_rad'], axis = 1)
y_lon = dataset.iloc[13750:86190, 10]
#y_lon = y_lon.values.reshape(len(y_lon), 1)
y_lon = y_lon.values.ravel()
y_lat = dataset.iloc[13750:86190, 11]
#y_lat = y_lat.values.reshape(len(y_lat), 1)
y_lat = y_lat.values.ravel()

# Feature Scaling
from sklearn.preprocessing import StandardScaler    # we should get values <-3, 3>
sc_X = StandardScaler()
sc_y_lon = StandardScaler()
sc_y_lat = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(X)) # not sure if we transform both X_train and X_test, but I would guess so
y_lon = pd.DataFrame(sc_y_lon.fit_transform(y_lon))
y_lat = pd.DataFrame(sc_y_lat.fit_transform(y_lat))
print(X)
print(y_lon)
print(y_lat)

# Splitting the dataset into the Training set and Test set
X_train = X.iloc[:65198, :]
y_lon_train = y_lon.iloc[:65198, :]
y_lat_train = y_lat.iloc[:65198, :]
X_test = X.iloc[65198:, :]
y_lon_test = y_lon.iloc[65198:, :]
y_lat_test = y_lat.iloc[65198:, :]

# Training the SVR model for longitude
from sklearn.svm import SVR
regressor_lon = SVR(kernel='rbf')
regressor_lon.fit(X_train, y_lon_train)   # train on X_train and y_train

# Training the SVR model for latitude
regressor_lat = SVR(kernel='rbf')
regressor_lat.fit(X_train, y_lat_train)   # train on X_train and y_train

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


# Decision Tree Regression


# # Importing the dataset
# dataset = pd.read_excel(r'C:\Users\mstep\Desktop\Bachelor Project\Data\output.xlsx')
# X = dataset.iloc[13750:86190, 2:]
# X = X.drop(['lon', 'lat', 'lon_rad', 'lat_rad'], axis = 1)
# y = dataset.iloc[13750:86190, 10:12]
#
# # Splitting the dataset into the Training set and Test set
# X_train = X.iloc[:65198, :]
# y_train = y.iloc[:65198, :]
# X_test = X.iloc[65198:, :]
# y_test = y.iloc[65198:, :]
#
# # Training the Decision Tree Regression model
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor()
# regressor.fit(X_train, y_train)
#
# # Predicting a new result
# y_pred = regressor.predict(X_test)
# y_pred = pd.DataFrame(y_pred)
#
# # Evaluating the model's performance
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# MAE = mean_absolute_error(y_test, y_pred)
# RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
# R2 = r2_score(y_test, y_pred)
# print(MAE)
# print(RMSE)
# print(R2)



# Random Forest Regression


# # Importing the dataset
# dataset = pd.read_excel(r'C:\Users\mstep\Desktop\Bachelor Project\Data\output.xlsx')
# X = dataset.iloc[13750:86190, 2:]
# X = X.drop(['lon', 'lat', 'lon_rad', 'lat_rad'], axis = 1)
# y = dataset.iloc[13750:86190, 10:12]
#
# # Splitting the dataset into the Training set and Test set
# X_train = X.iloc[:65198, :]
# y_train = y.iloc[:65198, :]
# X_test = X.iloc[65198:, :]
# y_test = y.iloc[65198:, :]
#
# # Training the Random Forest Regression model
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=10)  # here I would build one tree for each transit
# regressor.fit(X_train, y_train)
#
# # Predicting a new result
# y_pred = regressor.predict(X_test)
# y_pred = pd.DataFrame(y_pred)
#
# # Evaluating the model's performance
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# MAE = mean_absolute_error(y_test, y_pred)
# RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
# R2 = r2_score(y_test, y_pred)
# print(MAE)
# print(RMSE)
# print(R2)

# def domodel():
#     model = keras.Sequential(
#         [
#             layers.Dense(1, activation="relu", name="layer1"),
#             layers.Dense(3, activation="relu", name="layer2"),
#             layers.Dense(4, name="layer3"),
#
#         ]
#     )
#
#     modelRNN = keras.Sequential(
#         [
#             layers.Embedding(input_dim=1000, output_dim=64),
#             layers.LSTM(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.SimpleRNN(128),
#             layers.Flatten(),
#             layers.Dense(10, activation="relu")
#
#
#         ]
#     )
