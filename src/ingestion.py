import logging
import os
import pandas as pd
import sys

from datetime import datetime
from icecream import ic

from config import DATA_PATH, INPUT_FOLDER_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def merge_multiple_dataframe():
    """Function for ingesting data. Checks datasets,
        combines them, discards duplicates and writes
        metadata to ingestedfiles.txt and managed data to finaldata.csv
    """

    df = pd.DataFrame()
    file_names = []

    logging.info(f"Reading files from: {INPUT_FOLDER_PATH}.")

    for file in os.listdir(INPUT_FOLDER_PATH):
        ic(file)
        ic(INPUT_FOLDER_PATH)
        file_path = os.path.join(INPUT_FOLDER_PATH, file)
        df_tmp = pd.read_csv(file_path)
        ic(df_tmp)

        file = os.path.join(*file_path.split(os.path.sep)[-3:])
        ic(file)
        file_names.append(file)

        df = df.append(df_tmp, ignore_index=True)

    logging.info("Dropping duplicated lines.")
    df = df.drop_duplicates().reset_index(drop=1)

    logging.info("Saving the ingested metadata: ingestedfiles.txt.")
    with open(os.path.join(DATA_PATH, 'ingestedfiles.txt'), "w") as file:
        file.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))

    logging.info("Saving ingested data: finaldata.csv.")
    df.to_csv(os.path.join(DATA_PATH, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    logging.info("Running ingestion.py")
    merge_multiple_dataframe()
