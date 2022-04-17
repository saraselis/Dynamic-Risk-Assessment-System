import logging
import os
import pandas as pd
import pickle
import sys

from icecream import ic
from sklearn.metrics import f1_score

from config import MODEL_PATH, TEST_DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def score_model():
    """Load a trained model and test data and calculate an F1 assessment
        for the model in the test data and save the result in the lastscore.txt file
    """
    logging.info("Loading the testdata.csv file.")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    logging.info("Loading trained model.")
    model = pickle.load(open(os.path.join(MODEL_PATH, 'trainedmodel.pkl'), 'rb'))

    logging.info("Preparing the test data.")
    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting the test data")
    y_pred = model.predict(X_df)
    f1 = f1_score(y_true, y_pred)
    logging.info(f"F1 score: {f1}")
    ic(f1)

    with open(os.path.join(MODEL_PATH, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1}")


if __name__ == '__main__':
    logging.info("Running scoring.py")
    score_model()
