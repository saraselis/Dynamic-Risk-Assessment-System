import pandas as pd
import re
import subprocess

from flask import Flask, jsonify, request

import diagnostics


# Set up variables for use in our script
app_ = Flask(__name__)
app_.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


@app_.route('/')
def index():
    return "Opa eai"


@app_.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """Prediction endpoint that loads data given the file path
        and calls the prediction.

    Returns:
        json: model predictions
    """
    filepath = request.get_json()['filepath']

    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)

    preds = diagnostics.model_predictions(df)
    return jsonify(preds.tolist())


@app_.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """Scoring endpoint that runs the script scoring and
        gets the score of the deployed model.

    Returns:
        str: model f1 score
    """
    output = subprocess.run(['python', 'scoring.py'],
                            capture_output=True).stdout
    output = re.findall(r'f1 score = \d*\.?\d+', output.decode())[0]
    return output


@app_.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """Summary statistics endpoint that calls dataframe summary
        function from diagnostics.py

    Returns:
        json: summary statistics
    """
    return jsonify(diagnostics.dataframe_summary())


@app_.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diag():
    missing = diagnostics.missing_percentage()
    time = diagnostics.execution_time()
    outdated = diagnostics.outdated_packages_list()

    ret = {
        'missing_percentage': missing,
        'execution_time': time,
        'outdated_packages': outdated
    }

    return jsonify(ret)


if __name__ == "__main__":
    app_.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
