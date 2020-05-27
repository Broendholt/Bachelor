import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier
import geopy
from geopy.distance import VincentyDistance

def main():

    # model = machine learning model to use
    # 1 - Decision tree
    # 2 - Random forrest
    # 3 - Linear Regression
    # 4 - Support Vector Regression
    # 5 - Multi Layer Perceptron

    # test:
    # 1 = SOG (speed over ground)
    # 2 = HDG (heading)


    model_names = ['Decision tree', 'Random forrest', 'Linear Regression',
                   'Support Vector Machines', 'Multi Layer Perceptron']

    # model = 1
    test = 2
    for model in range(1,6):
        for i in range(1, 12):

            if i == 11:
                i = -1

            print(model_names[model - 1])
            MAE, RMSE, R2 = run_model(model, test, False, False, i)
            print('Feature amount', i)
            print('MAE :', MAE)
            print('RMSE:', RMSE)
            print('R2  :', R2)
            print('')


def run_model(model, test, show_plot, show_print, n_best_features):
    col = ''
    axis_name = ''
    title = ''
    predictions = []
    MAE = 0
    RMSE = 0
    R2 = 0

    if test == 1:
        col = 'SOG'
        axis_name = 'Speed over ground'
    elif test == 2:
        col = 'HDG'
        axis_name = 'Heading in degrees'

    data_set = feature_selection(col, n_best_features, show_print)

    X_train, y_train, X_test, y_test = prep_data(data_set, col, 10)

    if model == 1:
        # 1 - Decision tree
        predictions, MAE, RMSE, R2 = decision_tree(X_train, y_train, X_test, y_test, show_print)
        title = 'Decision Tree'

    elif model == 2:
        # 2 - Random forrest
        predictions, MAE, RMSE, R2 = random_forrest(X_train, y_train, X_test, y_test, show_print)
        title = 'Random Forrest'

    elif model == 3:
        # 3 - Linear Regression
        predictions, MAE, RMSE, R2 = linear_regression(X_train, y_train, X_test, y_test, show_print)
        title = 'Linear Regression'

    elif model == 4:
        # 4 - Support Vector Machines
        predictions, MAE, RMSE, R2 = support_vector_machines(X_train, y_train, X_test, y_test, show_print)
        title = 'Support Vector Machines'

    elif model == 5:
        # 5 - Multi Layer Perceptron
        predictions, MAE, RMSE, R2 = multi_layer_perceptron(X_train, y_train, X_test, y_test, show_print)
        title = 'Multi Layer Perceptron'

    # Plot data
    if show_plot:
        plot_results(predictions, y_test, title, 'Seconds', axis_name)

    return MAE, RMSE, R2


def feature_selection(column_name, n_best, show_print):
    data_set = pd.read_excel('output.xlsx')

    column = column_name
    drop_table = ['lon', 'lat', 'lon_rad', 'lat_rad', 'datetime', 'OffCource', 'OffTrack', column]
    X = data_set.drop(drop_table, axis=1)

    y = data_set.drop(drop_table[:-1], axis=1)
    corr_pearson = X.corrwith(y[column], method="pearson")

    if n_best == -1:
        sorted = corr_pearson.sort_values()
    else:
        sorted = corr_pearson.sort_values()[-n_best:]

    new_filter_table = []

    for i in range(len(sorted)):
        if show_print:
            print(sorted[i])
        new_filter_table.append(sorted.index[i])

    new_filter_table.append(column)
    y = y.filter(new_filter_table)



    return y


def prep_data(data, column_to_predict, train_percent):
    # percent = (100 - train_percent) / 100
    # cut = int(len(data) * percent)

    cut = 55001

    X_train = data[:cut].drop([column_to_predict], axis=1)
    y_train = data[:cut].filter([column_to_predict], axis=1)

    X_test = data[cut:].drop([column_to_predict], axis=1)
    y_test = data[cut:].filter([column_to_predict], axis=1)

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


def plot_results(predictions, true_values, title_name, x_label, y_label):
    index = np.arange(len(true_values))

    true_patch = mpatches.Patch(color='red', label='Known Values')
    pred_patch = mpatches.Patch(color='blue', label='Predicted Values')

    plt.legend(handles=[true_patch, pred_patch])
    plt.plot(index, true_values, 'r', index, predictions, 'b')
    plt.title(title_name)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.show()


def rad_to_coord(rad):
    rad_to_deg = (180 / 3.14159265359) * rad

    degrees = int(rad_to_deg)
    print(degrees)

    min = str(rad_to_deg).split('.')[1]
    min = float('.' + min)
    min *= 60
    tmp = min
    min = str(min).split('.')[0]

    sec = str(tmp).split('.')[1]
    sec = float('.' + sec)
    sec *= 60

    if float(sec) >= 60:
        sec = 0

    return degrees, min, sec


def decision_tree(X_train, y_train, X_test, y_test, _print):
    from sklearn.tree import DecisionTreeRegressor

    regressor = DecisionTreeRegressor(max_depth=10)
    regressor.fit(X_train, y_train)

    # Predicting a new result
    y_pred = regressor.predict(X_test)
    y_pred = np.array(y_pred)

    # Evaluating the model's performance
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score

    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    if _print:
        print('Mean Absolute Error:', MAE)
        print('Root Mean Square Deviation:', RMSE)
        print('R2 Score:', R2)

    return y_pred, MAE, RMSE, R2


def random_forrest(X_train, y_train, X_test, y_test, _print):
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

    if _print:
        print('Mean Absolute Error:', MAE)
        print('Root Mean Square Deviation:', RMSE)
        print('R2 Score:', R2)

    return y_pred, MAE, RMSE, R2


def linear_regression(X_train, y_train, X_test, y_test, _print):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    y_pred = np.array(y_pred)

    # Evaluating the model's performance

    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    if _print:
        print('Mean Absolute Error:', MAE)
        print('Root Mean Square Deviation:', RMSE)
        print('R2 Score:', R2)

    return y_pred, MAE, RMSE, R2


def support_vector_machines(X_train, y_train, X_test, y_test, _print):
    from sklearn.svm import SVR

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)

    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)

    # Predicting a new result for longitude
    y_pred = sc_y.inverse_transform(regressor.predict(X_test))  # rescaled predictions
    y_pred = pd.DataFrame(y_pred)

    # Evaluating the model's performance
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    if _print:
        print('Mean Absolute Error:', MAE)
        print('Root Mean Square Deviation:', RMSE)
        print('R2 Score:', R2)

    return y_pred, MAE, RMSE, R2


def multi_layer_perceptron(X_train, y_train, X_test, y_test, _print):

    sc_X_train = StandardScaler()
    sc_X_test = StandardScaler()
    sc_y_train = StandardScaler()
    sc_y_test = StandardScaler()

    X_train = sc_X_train.fit_transform(X_train)
    X_test = sc_X_test.fit_transform(X_test)
    y_train = sc_y_train.fit_transform(y_train)
    y_test = sc_y_test.fit_transform(y_test)

    size = X_train.shape[1] + 0.1
    size = round(size)

    size_half = (size / 2) + 0.1
    size_half = round(size_half)

    print(size)
    print(size_half)

    # Defining the model
    model = Sequential()
    model.add(Dense(size, input_dim=size, activation='sigmoid'))
    model.add(Dense(size_half, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(Adam(lr=0.003), 'mean_squared_error')

    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)

    history = model.fit(X_train, y_train, epochs=45, validation_split=0.2, verbose=0, callbacks=[earlystopper])

    # Plots 'history'
    # history_dict = history.history
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']
    # plt.plot(loss_values, 'bo', label='training loss')
    # plt.plot(val_loss_values, 'r', label='training loss val')
    # plt.show()

    # Runs model with its current weights on the training and testing data
    y_pred = sc_y_test.inverse_transform(model.predict(X_test))
    y_test = sc_y_test.inverse_transform(y_test)

    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    if _print:
        print('Mean Absolute Error:', MAE)
        print('Root Mean Square Deviation:', RMSE)
        print('R2 Score:', R2)

    return y_pred, MAE, RMSE, R2


main()
