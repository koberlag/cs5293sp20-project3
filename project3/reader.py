
import os
import pandas as pd


PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath("app.py")), 'project3')

def read_yummly_json(): 
    """Load the yummly.json data into a pandas dataframe and return"""

    print(PROJECT_DIR)
    df = pd.DataFrame()
    try:
        file_path = PROJECT_DIR + os.path.sep + "yummly.json"
        df = pd.read_json(file_path)
    except Exception as ex:
        print(ex)
    return df