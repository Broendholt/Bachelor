import numpy as np
import matplotlib.pyplot as plt
import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd


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


show_data_in_plot(lon)
show_data_in_plot(lat)


