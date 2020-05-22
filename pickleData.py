import pickle
from os import listdir
from os.path import join, splitext
import pandas as pd
import time

data_folder = r'C:\Users\cbroe\OneDrive\Skrivebord\Stuff\School\bachelor\data_2(Berlin)'

data_files = listdir(data_folder)


data = []



for i in range(len(data_files)):
    tic = time.perf_counter()

    print(i, "/", 260)
    data_path = join(data_folder, data_files[i])
    example_dict = pd.read_pickle(data_path)
    example_dict.to_excel("output1.xlsx")

    dataTmp = pd.read_excel("output1.xlsx")
    dataTmp.set_index('datetime')
    data.append(dataTmp)

    toc = time.perf_counter()
    timeDelta = (toc - tic) * (260 - 1)
    print(f'ESTIMATED TIME: {timeDelta / 60:0.4f} MIN')
    print("---------------------------------------------")


prev_columns = ""

data_combined = []
data_tmp = []

periods = 86400 / 1
index_range = pd.date_range(start='2019-10-23 00:00:00', end='2019-10-24 00:00:00', periods=periods)
df = pd.DataFrame(columns=['datetime'])
df['datetime'] = index_range
df['datetime'] = df.datetime.dt.round(freq='s')


data[0].set_index('datetime')
data[0]['datetime'] = data[0].datetime.dt.round(freq='s')
tmp = df.merge(data[0], 'outer', on='datetime')

for i in range(len(data) - 1):
    tic = time.perf_counter()
    print(i, "/", 260)
    data[i + 1].set_index('datetime')
    data[i + 1]['datetime'] = data[i + 1].datetime.dt.round(freq='s')
    tmp = tmp.combine_first(data[i + 1])
    toc = time.perf_counter()
    timeDelta = (toc - tic) * (260 - 1)
    print(f'ESTIMATED TIME: {timeDelta / 60:0.4f} MIN')
    print("---------------------------------------------")

tmp.to_excel('output2.xlsx')

print(tmp)


for i in range(len(data)):
    columns = data[i].columns[0]
    print(columns)
    print(data[i].size)


