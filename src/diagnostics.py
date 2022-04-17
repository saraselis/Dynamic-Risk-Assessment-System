import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
import sys
import timeit

from config import DATA_PATH, TEST_DATA_PATH, PROD_DEPLOYMENT_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def model_predictions(X_df: 'pd.DataFrame') -> 'np.array':
    """Load the deployed model and do predictions

    Args:
        X_df (pd.DataFrame): Dataframe with features

    Returns:
        y_pred (np.array): Predictions
    """

    logging.info("Loading the deployed model.")
    model = pickle.load(open(os.path.join(PROD_DEPLOYMENT_PATH, 'trainedmodel.pkl'), 'rb'))

    logging.info("Predictions")
    y_pred = model.predict(X_df)
    return y_pred


def dataframe_summary() -> list:
    """
    Loads finaldata.csv and calculates mean, median and std
    on numerical data.

    Returns:
        statistics_dict (list[dict]): Each dict contains column name,
        mean, median and std.
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')

    logging.info("Calculating statistics for data")
    statistics_dict = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        std = data_df[col].std()

        statistics_dict[col] = {'mean': mean, 'median': median, 'std': std}

    return statistics_dict


def missing_percentage() -> list:
    """
    Calculates percentage of missing data for each column
    in finaldata.csv

    Returns:
        missing_list (list[dict]): Each dict contains column name and percentage.
    """
    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    # data_df = data_df.drop(['corporation', 'exited'], axis=1)

    logging.info("Calculating missing data percentage")
    missing_list = {col: {'percentage': perc} for col, perc in zip(
        data_df.columns, data_df.isna().sum() / data_df.shape[0] * 100)}

    return missing_list


def _ingestion_timing():
    """
    Runs ingestion.py script and measures execution time

    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing


def _training_timing():
    """
    Runs training.py script and measures execution time

    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing


def execution_time():
    """
    Gets average execution time for data ingestion and model training
    by running each 25 times

    Returns:
        list[dict]: mean of execution times for each script
    """
    logging.info("Calculating time for ingestion.py")
    ingestion_time = []
    for _ in range(20):
        time = _ingestion_timing()
        ingestion_time.append(time)

    logging.info("Calculating time for training.py")
    training_time = []
    for _ in range(20):
        time = _training_timing()
        training_time.append(time)

    ret_list = [
        {'ingest_time_mean': np.mean(ingestion_time)},
        {'train_time_mean': np.mean(training_time)}
    ]

    return ret_list


def outdated_packages_list():
    """
    Check dependencies status from requirements.txt file using pip-outdated
    which checks each package status if it is outdated or not

    Returns:
        str: stdout of the pip-outdated command
    """
    logging.info("Checking outdated dependencies")
    dependencies = subprocess.run('pip-outdated requirements.txt',
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, encoding='utf-8')

    dep = dependencies.stdout
    dep = dep.translate(str.maketrans('', '', ' \t\r'))
    dep = dep.split('\n')
    dep = [dep[3]] + dep[5:-3]
    dep = [s.split('|')[1:-1] for s in dep]

    return dep


if __name__ == '__main__':

    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))
    X_df = test_df.drop(['corporation', 'exited'], axis=1)

    logging.info("Model predictions on testdata.csv:", model_predictions(X_df))

    logging.info(f"Summary statistics: {json.dumps(dataframe_summary(), indent=4)}")

    logging.info(f"Missing percentage: {json.dumps(missing_percentage(), indent=4)}")

    logging.info(f"Execution time: {json.dumps(execution_time(), indent=4)}")

    logging.info("Outdated Packages")
    dependencies = outdated_packages_list()
    for row in dependencies:
        logging.info('f deps: {:<20}{:<10}{:<10}{:<10}'.format(*row))
