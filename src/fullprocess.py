import logging
import pandas as pd
import os
import re
import sys

from sklearn.metrics import f1_score

import scoring
import training
import ingestion
import reporting
import deployment
import diagnostics

from config import INPUT_FOLDER_PATH, PROD_DEPLOYMENT_PATH, DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    logging.info("Checking for new data")

    logging.info("Read ingestedfiles.txt")
    with open(os.path.join(PROD_DEPLOYMENT_PATH, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    source_files = set(os.listdir(INPUT_FOLDER_PATH))

    if len(source_files.difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None

    logging.info('Ingesting new data')
    ingestion.merge_multiple_dataframe()

    logging.info("Checking for model drift")

    with open(os.path.join(PROD_DEPLOYMENT_PATH, "latestscore.txt")) as file:
        deployed_score = re.findall(r'\d*\.?\d+', file.read())[0]
        deployed_score = float(deployed_score)

    data_df = pd.read_csv(os.path.join(DATA_PATH, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    y_pred = diagnostics.model_predictions(X_df)
    new_score = f1_score(y_df.values, y_pred)

    logging.info(f"Deployed score = {deployed_score}\nNew score = {new_score}")

    logging.info("Checking if model drifted")
    if(new_score >= deployed_score):
        logging.info("No model drift occurred")
        return None

    logging.info("Re-training model")
    training.train_model()

    logging.info("Re-scoring model")
    scoring.score_model()

    logging.info("Re-deploying model")
    deployment.deploy_model()

    logging.info("Running diagnostics and reporting...")

    logging.info("Run diagnostics.py and reporting.py for the re-deployed model")
    reporting.plot_confusion_matrix()
    reporting.generate_pdf_report()
    os.system("python apicalls.py")


if __name__ == '__main__':
    main()
