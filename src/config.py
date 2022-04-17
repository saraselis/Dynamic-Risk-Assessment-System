import json
import os

from icecream import ic


# Load config.json and correct path variable
with open('config.json', 'r') as file:
    CONFIG = json.load(file)

INPUT_FOLDER_PATH = os.path.join(os.path.abspath('../Dynamic-Risk-Assessment-System/'),'data', CONFIG['input_folder_path'])
ic(INPUT_FOLDER_PATH)

DATA_PATH = os.path.join(os.path.abspath('../Dynamic-Risk-Assessment-System/'), 'data', CONFIG['output_folder_path'])
ic(DATA_PATH)

TEST_DATA_PATH = os.path.join(os.path.abspath('../Dynamic-Risk-Assessment-System/'), 'data', CONFIG['test_data_path'])

MODEL_PATH = os.path.join(os.path.abspath('../Dynamic-Risk-Assessment-System/'), 'model', CONFIG['output_model_path'])

PROD_DEPLOYMENT_PATH = os.path.join(os.path.abspath('../Dynamic-Risk-Assessment-System/'), 'model', CONFIG['prod_deployment_path'])

