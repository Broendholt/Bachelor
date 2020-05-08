import socket


def get_user_data():

    # gets the name og the host pc/machine
    pc_name = socket.gethostname()

    if pc_name == "LAPTOP-KV41CBJV":
        return r"C:\Users\cbroe\OneDrive\Skrivebord\Stuff\School\bachelor"
    elif pc_name is "":
        return None


# variables for data selection
data_structure = {"data_1":
                      {"data_type": "excel",
                       "data_extension": "xlsx",
                       "data_description": "no desc yet"},

                  "data_2(Berlin)":
                      {"data_type": "pickle",
                       "data_extension": "pkl",
                       "data_description": "no desc yet"},

                  "Data_3(Prins_R)":
                      {"data_type": "csv",
                       "data_extension": "csv",
                       "data_description": "no desc yet"}}
