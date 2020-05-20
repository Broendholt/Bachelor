import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd
from datetime import date, datetime, timedelta

def read_excel(file_name):

    data_path = cf.get_user_data()
    data_folder_path = join(data_path, 'data_1')

    data_files = listdir(data_folder_path)

    data = []

    if file_name is 'all':
        print('Read all')
        for i in range(len(data_files)):
            if splitext(data_files[i])[1] == '.xlsx':
                data.append(pd.read_excel(join(data_folder_path, data_files[i])))
                # data.append(pd.read_excel(join(data_folder_path, data_files[i]), index_col=0))

    elif type(file_name) is list:
        print('Read list')
        # read only the specified files (multiple)
        return
    elif type(file_name) is str:
        print('Read one')
        # read only the specified file (one)
        return

    return data


data_excel = read_excel('all')

periods = 86400 / 1
index_range = pd.date_range(start='2018-12-27 00:00:00', end='2018-12-28 00:00:00', periods=periods)
df = pd.DataFrame(columns=['datetime'])
df['datetime'] = index_range
df['datetime'] = df.datetime.dt.round(freq='s')


data_excel[0]['datetime'] = data_excel[0].datetime.dt.round(freq='s')

tmp = df.merge(data_excel[0], 'outer', on='datetime')


for i in range(len(data_excel) - 1):
    data_excel[i + 1]['datetime'] = data_excel[i + 1].datetime.dt.round(freq='s')
    tmp = tmp.merge(data_excel[i + 1], 'outer', on='datetime')


print("start interpolation")

prev_index = 0
prev_val = 0

curr_index = 0
curr_val = 0

columns = tmp.columns
for c in range(len(columns)):

    columns1 = tmp.columns
    count = 0
    for j in range(len(columns1)):
        count += tmp[columns1[j]].isna().sum()

    print(c, "Column:", columns[c], "NaN value:", count)

    column = columns[c]
    if column == 'lon':
        continue
    if column == 'lat':
        continue

    for index, row in tmp.iterrows():

        val1 = 0
        val2 = 0

        prev_index = 0

        if index < prev_index:
            continue

        if str(tmp.at[index, column]) != 'nan':
            val1 = tmp.at[index, column]
            prev_index = index
            found_val = False

            while not found_val:
                prev_index += 1
                if prev_index >= len(tmp):
                    val2 = 0
                    break

                if str(tmp.at[prev_index, column]) != 'nan':
                    val2 = tmp.at[prev_index, column]

                    for j in range(0, prev_index - index):

                        val = val2 - val1
                        valp = val / (prev_index - index)

                        if index + j >= len(tmp):
                            break

                        tmp.at[index + j, column] = val1 + (valp * (j))

                    found_val = True




'''
for i in range(10):

    columns = tmp.columns
    count = 0
    for j in range(len(columns)):
        count += tmp[columns[j]].isna().sum()

    print(i, ":", count)
    tmp.interpolate(method='index', axis=1, inplace=True)


print("interpolation ended")

'''
print(tmp)

tmp.to_excel("output.xlsx")
