import diagnostics
import logging
import pandas as pd
import os
import sys

from reportlab.lib.colors import green, lavender, red
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from sklearn.metrics import confusion_matrix

from config import DATA_PATH, MODEL_PATH, TEST_DATA_PATH


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def plot_confusion_matrix():
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    y_true = test_df.pop('exited')
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting teste data.")
    y_pred = diagnostics.model_predictions(X_df)

    logging.info("Plotting and saving the confusion matrix.")
    fig, ax = confusion_matrix(y_true, y_pred)

    # fig.savefig(os.path.join(MODEL_PATH, 'confusionmatrix.png'))


def _get_statistics_df() -> 'pd.DataFrame':
    stats = diagnostics.dataframe_summary()
    missing = diagnostics.missing_percentage()

    data = {'Column Name': [k for k in missing.keys()]}
    data['Missing %'] = [missing[column]['percentage']
                         for column in data['Column Name']]

    temp_col = list(stats.keys())[0]
    for stat in stats[temp_col].keys():
        data[stat] = [round(stats[column][stat], 2) if stats.get(column,None)
                        else '-' for column in data['Column Name']]

    return data


def generate_pdf_report():
    pdf = canvas.Canvas(os.path.join(MODEL_PATH, 'summary_report.pdf'),
                        pagesize=A4)

    pdf.setTitle("Model Summary Report.")

    pdf.setFontSize(24)
    pdf.setFillColorRGB(31 / 256, 56 / 256, 100 / 256)
    pdf.drawCentredString(300, 800, "Model Summary Report")

    logging.info("REPORT: Ingest data section")
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 750, "Ingested Data")

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 725, "List of files used:")


    logging.info("REPORT: Ingested files")

    with open(os.path.join(DATA_PATH, "ingestedfiles.txt")) as file:
        pdf.setFontSize(12)
        text = pdf.beginText(40, 705)
        text.setFillColor('black')

        for line in file.readlines():
            text.textLine(line.strip('\n'))

        pdf.drawText(text)

    logging.info("REPORT: Data statistics and missing percentage")

    data = _get_statistics_df()
    data_df = pd.DataFrame(data)
    data_table = data_df.values.tolist()
    data_table.insert(0, list(data_df.columns))

    logging.info("REPORT: Draw summary table")
    stats_table = Table(data_table)
    stats_table.setStyle([('GRID', (0, 0), (-1, -1), 1, 'black'),
                          ('BACKGROUND', (0, 0), (-1, 0), lavender)])

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 645, "Statistics Summary")

    stats_table.wrapOn(pdf, 40, 520)
    stats_table.drawOn(pdf, 40, 520)

    logging.info("REPORT: Trained model section")
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 490, "Trained Model Scoring on Test Data")

    pdf.setFontSize(12)
    pdf.setFillColorRGB(128 / 256, 128 / 256, 128 / 256)
    pdf.drawString(25, 480, "testdata.csv")

    logging.info("REPORT: Model score")
    with open(os.path.join(MODEL_PATH, "latestscore.txt")) as file:
        pdf.setFontSize(12)
        pdf.setFillColor('black')
        pdf.drawString(40, 460, file.read())

    logging.info("REPORT: Model confusion matrix")
    pdf.drawInlineImage(os.path.join(MODEL_PATH,'confusionmatrix.png'),
                        40, 150, width=300, height=300)

    logging.info("REPORT: New page")
    pdf.showPage()

    logging.info("REPORT: Diagnostics section")
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 780, "Diagnostics")

    logging.info("REPORT: Execution time")
    timings = diagnostics.execution_time()

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 755, "Execution times:")

    pdf.setFontSize(12)
    text = pdf.beginText(40, 735)
    text.setFillColor('black')

    for time in timings:
        for k, v in time.items():
            text.textLine(f"{k} = {round(v, 4)}")

    pdf.drawText(text)

    logging.info("REPORT: Draw outdated dependencies table")
    data = diagnostics.outdated_packages_list()

    table_style = TableStyle()
    table_style.add('GRID', (0, 0), (-1, -1), 1, 'black')
    table_style.add('BACKGROUND', (0, 0), (-1, 0), lavender)

    for row, values in enumerate(data[1:], start=1):
        if(values[1] != values[2]):
            table_style.add('TEXTCOLOR', (1, row), (1, row), red)
            table_style.add('TEXTCOLOR', (2, row), (2, row), green)

    depend_table = Table(data)
    depend_table.setStyle(table_style)

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 690, "Outdated Dependencies")

    depend_table.wrapOn(pdf, 40, 325)
    depend_table.drawOn(pdf, 40, 325)

    pdf.save()


if __name__ == '__main__':
    logging.info("Running: reporting.py")

    logging.info("Generating: confusion matrix")
    plot_confusion_matrix()

    logging.info("Generating PDF report")
    generate_pdf_report()
