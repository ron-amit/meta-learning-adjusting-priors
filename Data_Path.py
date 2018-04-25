import os

def get_data_path():
    # The path of the directory in which raw data is saved:

    # base_dir = os.path.expanduser("~") # user home dir
    base_dir = os.path.dirname(os.getcwd())  # code dir

    data_path = os.path.join(base_dir, 'ML_data_sets')
    return data_path

