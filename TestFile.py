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

    # # model = 1
    # test = 2
    # for model in range(1,6):
    #     for i in range(1, 12):
    #
    #         if i == 11:
    #             i = -1
    #
    #         print(model_names[model - 1])
    #         MAE, RMSE, R2 = run_model(model, test, False, False, i)
    #         print('Feature amount', i)
    #         print('MAE :', MAE)
    #         print('RMSE:', RMSE)
    #         print('R2  :', R2)
    #         print('')

    #for i in range(1, 6):
    #    create_new_plots(-1, i)


    data_set = pd.read_excel('test_All_delta5NEW.xlsx')

    pred_delta_lon = data_set.filter(['pred_lon']).to_numpy()
    pred_delta_lat = data_set.filter(['pred_lat']).to_numpy()
    true_delta_lon = data_set.filter(['lon_true_delta']).to_numpy()
    true_delta_lat = data_set.filter(['lat_true_delta']).to_numpy()
    #
    lon_true = data_set.filter(['lon_true']).to_numpy()
    lat_true = data_set.filter(['lat_true']).to_numpy()
    #
    lon_new = []
    lat_new = []
    #

    #
    print(pred_delta_lon)
    #
    lon_start = lon_true[0]
    lat_start = lat_true[0]

    for i in range(len(data_set)):

        pred_delta_lon_val = pred_delta_lon[i] / 1000000
        pred_delta_lat_val = pred_delta_lat[i] / 1000000

        lon_new.append(lon_start - pred_delta_lon_val)
        lat_new.append(lat_start - pred_delta_lat_val)

        lon_start = lon_start - pred_delta_lon_val
        lat_start = lat_start - pred_delta_lat_val
    #
    #
    true_patch = mpatches.Patch(color='red', label='Predicted Values')
    pred_patch = mpatches.Patch(color='blue', label='Known Values')
    index = np.arange(0, len(lon_true))

    plt.subplot(211)
    plt.xlabel('Seconds')
    plt.ylabel('Latitude')
    plt.title('Multi Layer Perceptron - all features')
    plt.legend(handles=[true_patch, pred_patch])

    plt.plot(index, lat_true, 'b', index, lat_new, 'r--')

    plt.subplot(212)
    plt.xlabel('Seconds')
    plt.ylabel('Longitude')
    plt.legend(handles=[true_patch, pred_patch])

    plt.plot(index, lon_true, 'b', index, lon_new, 'r--')


    # cut_s = 1000
    # cut_e = 5000
    # plt.plot(lat_true[cut_s:cut_e], lon_true[cut_s:cut_e], 'b', lat_new[cut_s:cut_e], lon_new[cut_s:cut_e], 'r')

    plt.show()

    return

    """
    data_set_sog = feature_selection('SOG', -1, False)
    X_train_sog, y_train_sog, X_test_sog, y_test_sog = prep_data(data_set_sog, 'SOG', -1)

    data_set_hdg = feature_selection('HDG', -1, False)
    X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg = prep_data(data_set_hdg, 'HDG', -1)

    if i == 1:
        pred_sog, MAE, RMSE, R2 = decision_tree(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        pred_hdg, MAE, RMSE, R2 = decision_tree(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)

    elif i == 2:
        pred_sog, MAE, RMSE, R2 = random_forrest(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        pred_hdg, MAE, RMSE, R2 = random_forrest(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)

    elif i == 3:
        pred_sog, MAE, RMSE, R2 = linear_regression(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        pred_hdg, MAE, RMSE, R2 = linear_regression(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)

    elif i == 4:
        pred_sog, MAE, RMSE, R2 = support_vector_machines(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        pred_hdg, MAE, RMSE, R2 = support_vector_machines(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)

    elif i == 5:
        pred_sog, MAE, RMSE, R2 = multi_layer_perceptron(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        pred_hdg, MAE, RMSE, R2 = multi_layer_perceptron(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)

    index = np.arange(0, len(pred_sog))

    true_patch = mpatches.Patch(color='red', label='Predicted Values')
    pred_patch = mpatches.Patch(color='blue', label='Known Values')

    plt.subplot(211)
    plt.xlabel('Index')
    plt.ylabel('Speed over ground (SOG)')
    plt.title('Multi Layer Perceptron - all features')
    plt.legend(handles=[true_patch, pred_patch])
    plt.plot(index, pred_sog, 'r', index, y_test_sog, 'b')

    plt.subplot(212)
    plt.xlabel('Index')
    plt.ylabel('Heading (HDG)')
    plt.legend(handles=[true_patch, pred_patch])
    plt.plot(index, pred_hdg, 'r', index, y_test_hdg, 'b')

    plt.show()
    """



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

    lon_test, lat_test, X_train, y_train, X_test, y_test = prep_data(data_set, col, 10)

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
    if column_name == 'Lat_delta':
        drop_table = ['lon', 'lat', 'datetime', 'lon_rad', 'WindDirectionTrue', 'WindDirectionRel', 'WindSpeedRel', 'lat_rad', 'WindSpeedTrue', 'Lon_delta', 'OffCource', 'OffTrack', column]
    elif column_name == 'Lon_delta':
        drop_table = ['lon', 'lat', 'datetime', 'lon_rad', 'WindDirectionTrue', 'WindDirectionRel', 'WindSpeedRel', 'lat_rad', 'WindSpeedTrue', 'Lat_delta', 'OffCource', 'OffTrack', column]

    X = data_set.drop(drop_table, axis=1)

    y = data_set.drop(drop_table[:-1], axis=1)
    corr_pearson = X.corrwith(y[column], method="pearson")

    if n_best == -1:
        sorted = corr_pearson.sort_values()
    else:
        sorted = corr_pearson.sort_values()[-n_best:]

    print(sorted)

    new_filter_table = []

    for i in range(len(sorted)):
        if show_print:
            print(sorted[i])
        new_filter_table.append(sorted.index[i])

    new_filter_table.append(column)
    y = y.filter(new_filter_table)

    return y


def create_new_plots(n_features, model):

    data_set = pd.read_excel('output.xlsx')

    new_true_lon = []
    new_true_lat = []

    new_pred_lon = []
    new_pred_lat = []

    lon_true = data_set.filter(['lon_rad']).to_numpy()[55333:]
    lat_true = data_set.filter(['lat_rad']).to_numpy()[55333:]

    lon_true_delta = data_set.filter(['Lon_delta']).to_numpy()[55333:]
    lat_true_delta = data_set.filter(['Lat_delta']).to_numpy()[55333:]

    data_set_lon_delta = feature_selection('Lon_delta', -1, False)
    X_train_sog, y_train_sog, X_test_sog, y_test_sog = prep_data(data_set_lon_delta, 'Lon_delta', -1)

    data_set_lat_delta = feature_selection('Lat_delta', -1, False)
    X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg = prep_data(data_set_lat_delta, 'Lat_delta', -1)

    pred_lon = []
    pred_lat = []

    if model == 1:
        pred_lon, MAE, RMSE, R2 = decision_tree(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)
        pred_lat, MAE, RMSE, R2 = decision_tree(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)

    elif model == 2:
        pred_lon, MAE, RMSE, R2 = random_forrest(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)
        pred_lat, MAE, RMSE, R2 = random_forrest(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)

    elif model == 3:
        pred_lon, MAE, RMSE, R2 = linear_regression(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)
        pred_lat, MAE, RMSE, R2 = linear_regression(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)

    elif model == 4:
        pred_lon, MAE, RMSE, R2 = support_vector_machines(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)
        pred_lat, MAE, RMSE, R2 = support_vector_machines(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)

    elif model == 5:
        pred_lon, MAE, RMSE, R2 = multi_layer_perceptron(X_train_sog, y_train_sog, X_test_sog, y_test_sog, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)
        pred_lat, MAE, RMSE, R2 = multi_layer_perceptron(X_train_hdg, y_train_hdg, X_test_hdg, y_test_hdg, False)
        print("Model#", model)
        print(MAE)
        print(RMSE)
        print(R2)

    data = {'pred_lon': pred_lon.tolist()}

    df = pd.DataFrame(data)

    df['pred_lat'] = pred_lat
    df['y_test_sog'] = y_test_sog
    df['y_test_hdg'] = y_test_hdg
    df['lon_true'] = lon_true
    df['lat_true'] = lat_true
    df['lon_true_delta'] = lon_true_delta
    df['lat_true_delta'] = lat_true_delta

    df.to_excel("test_All_delta" + str(model) + "NEW.xlsx")

    return

    origin_lon = lon_true[0]
    origin_lat = lat_true[0]

    # True values ---------------------------------------------------------
    for i in range(1, len(y_test_sog)):

        lo = origin_lon
        la = origin_lat

        # if i <= len(new_true_lat) - 1:
        #     lo = new_true_lon[i] * (3.14159265359 / 180)
        #     la = new_true_lat[i] * (3.14159265359 / 180)

        lon_true_val, lat_true_val = get_coords(lo, la, y_test_sog[i], y_test_hdg[i])

        if lon_true_val <= 0.1 or lat_true_val <= 0.1:
            new_true_lon.append(-999)
            new_true_lat.append(-999)
        else:
            new_true_lon.append(lon_true_val)
            new_true_lat.append(lat_true_val)

        origin_lon = lon_true_val * (3.14159265359 / 180)
        origin_lat = lat_true_val * (3.14159265359 / 180)

    # Check for outliers in the true values
    for i in range(len(new_true_lon)):
        if new_true_lon[i] == -999:
            new_true_lon[i] = new_true_lon[i - 1]

        if new_true_lat[i] == -999:
            new_true_lat[i] = new_true_lat[i - 1]

    origin_lon = lon_true[0]
    origin_lat = lat_true[0]

    print(len(pred_sog), len(pred_hdg))
    # Pred values ---------------------------------------------------------
    for i in range(0, len(pred_sog - 1)):

        lo = origin_lon
        la = origin_lat

        # if i <= len(new_true_lat) - 1:
        #     lo = new_true_lon[i] * (3.14159265359 / 180)
        #     la = new_true_lat[i] * (3.14159265359 / 180)

        lon_pred_val, lat_pred_val = get_coords(lo, la, pred_sog[i], pred_hdg[i])

        if lon_pred_val <= 0.1 or lat_pred_val <= 0.1:
            new_pred_lon.append(-999)
            new_pred_lat.append(-999)
        else:
            new_pred_lon.append(lon_pred_val)
            new_pred_lat.append(lat_pred_val)

        origin_lon = lon_pred_val * (3.14159265359 / 180)
        origin_lat = lat_pred_val * (3.14159265359 / 180)

    # Check for outliers in the true values
    for i in range(len(new_pred_lon)):
        if new_pred_lon[i] == -999:
            new_pred_lon[i] = new_pred_lon[i - 1]

        if new_pred_lat[i] == -999:
            new_pred_lat[i] = new_pred_lat[i - 1]

    new_pred_lon = np.array(new_pred_lon)
    new_pred_lat = np.array(new_pred_lat)

    index = np.arange(0, len(pred_sog))
    index1 = np.arange(0, len(new_pred_lon))
    cut = 10000

    true_patch = mpatches.Patch(color='red', label='Predicted Values')
    pred_patch = mpatches.Patch(color='blue', label='Known Values')

    plt.subplot(211)
    plt.legend(handles=[true_patch, pred_patch])
    plt.plot(index1, new_pred_lon, 'r', index1, new_true_lon, 'b')

    plt.subplot(212)
    plt.legend(handles=[true_patch, pred_patch])
    plt.plot(index1, new_pred_lat, 'r', index1, new_true_lat, 'b')
    """
    plt.subplot(312)
    plt.plot(index, pred_sog, 'r', index, y_test_sog, 'b')

    plt.subplot(313)
    plt.plot(index, pred_hdg, 'r', index, y_test_hdg, 'b')

    plt.show(block=False)
    """
    plt.show()
    

def prep_data(data, column_to_predict, train_percent):
    # percent = (100 - train_percent) / 100
    # cut = int(len(data) * percent)

    cut = 55333

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

    min = str(rad_to_deg).split('.')[1]
    min = float('0.' + min.replace(']', '').replace('[', ''))
    min *= 60
    tmp = min
    min = str(min).split('.')[0]

    sec = str(tmp).split('.')[1]
    sec = float('0.' + sec)
    sec *= 60

    if float(sec) >= 60:
        sec = 0

    return float(str(str(degrees) + "." + str(min) + str(sec).replace('.', '')))

    return degrees, min, sec


def get_coords(origin_coord, speed, heading):
    origin = geopy.Point(origin_coord)
    new_coord = VincentyDistance(speed).destination(origin, heading)

    return new_coord


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

    print(regressor.coef_)

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

    y_pred = np.array(y_pred)


    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)

    if _print:
        print('Mean Absolute Error:', MAE)
        print('Root Mean Square Deviation:', RMSE)
        print('R2 Score:', R2)

    return y_pred, MAE, RMSE, R2


main()
