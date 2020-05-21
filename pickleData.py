import pickle as pkl
import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd

folder_path = r'C:\Users\cbroe\OneDrive\Skrivebord\Stuff\School\bachelor\data_2(Berlin)'
# path = r'C:\Users\cbroe\OneDrive\Skrivebord\Stuff\School\bachelor\data_2(Berlin)\transit_GPGLL_10_22_2019_022853.pkl'

data_files = listdir(folder_path)

data = []
prev_columns = ""

for i in range(len(data_files)):
    path = join(folder_path, data_files[i])

    pickle_in = open(path, "rb")
    example_dict = pkl.load(pickle_in)

    if prev_columns == example_dict.columns[0]:
        continue
    else:
        for j in range(len(example_dict.columns)):
            print(example_dict.columns[j])

        prev_columns = example_dict.columns[0]
        print("------------------------------------")




