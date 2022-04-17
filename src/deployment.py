import logging
import os
import shutil
import sys

from icecream import ic

from config import DATA_PATH, MODEL_PATH, PROD_DEPLOYMENT_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def deploy_model():
    """Copy the latest pickle file, the latestscore value,
        and the ingestfiles.txt file from the deployment directory
    """
    logging.info("Deploying trained model to production")

    shutil.copy(os.path.join(DATA_PATH, 'ingestedfiles.txt'), PROD_DEPLOYMENT_PATH)
    
    shutil.copy(os.path.join(MODEL_PATH, 'trainedmodel.pkl'), PROD_DEPLOYMENT_PATH)
    
    shutil.copy(os.path.join(MODEL_PATH, 'latestscore.txt'), PROD_DEPLOYMENT_PATH)


if __name__ == '__main__':
    logging.info("Running deployment.py")
    deploy_model()
