import ConfigFile as cf
from os import listdir
from os.path import join, splitext
import pandas as pd

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
        print("Found only " + i + " folders out of 3")


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




