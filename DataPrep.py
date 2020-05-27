
import pandas as pd

# Preps the data ie, puts it into the right size and removes nan / null values
# Returns full data set, x and y
def prepare_data(data_set, row_start, row_end, steps, percentile_for_training, excel_file):

    lon_name = 'lon_rad'
    lat_name = 'lat_rad'

    lon_extra_name = 'lon'
    lat_extra_name = 'lat'

    if excel_file == 2:
        lat_name = 'Lat'
        lon_name = 'Long'
        lon_extra_name = 'E_W'
        lat_extra_name = 'N_S'

    cut = (int)((row_end - row_start) / percentile_for_training)

    print('row_start:', row_start)
    print('row_end:', row_end)
    print('steps:', steps)
    print('cut:', cut)

    new_data_set = data_set.iloc[row_start:row_end:steps]
    x_lon = data_set[lon_name].iloc[row_end - cut:row_end:steps].to_numpy()
    x_lat = data_set[lat_name].iloc[row_end - cut:row_end:steps].to_numpy()

    y = new_data_set.drop([lat_name, lon_name, lon_extra_name, lat_extra_name], axis=1)
    y = y.iloc[row_start:row_end - cut:steps]
    return new_data_set, x_lon, x_lat, y


def prepare_data_feature_selection(data_set, row_start, row_end, steps, excel_file):

    lon_name = 'lon_rad'
    lat_name = 'lat_rad'

    lon_extra_name = 'lon'
    lat_extra_name = 'lat'

    if excel_file == 2:
        lat_name = 'Lat'
        lon_name = 'Long'
        lon_extra_name = 'E_W'
        lat_extra_name = 'N_S'

    print('row_start:', row_start)
    print('row_end:', row_end)
    print('steps:', steps)

    new_data_set = data_set.iloc[row_start:row_end:steps]
    x_lon = data_set[lon_name].iloc[row_start:row_end:steps]
    x_lat = data_set[lat_name].iloc[row_start:row_end:steps]

    y = data_set.iloc[row_start:row_end:steps]

    y = y.drop([lat_name, lon_name, lon_extra_name, lat_extra_name, 'datetime'], axis=1)
    y = y.drop(y.columns[y.columns.str.contains('unnamed', case=False)], axis=1)

    return new_data_set, x_lon, x_lat, y


# print(d)
# print(x_1)
# print(x_2)
# print(y)
