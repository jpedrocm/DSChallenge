# NubankChallenge
Software for creating predictive models about client financial default


# Prerequisites
pip
Python 2.7
virtualenv


# Installation
1) Open the command line
2) Install the prerequisites
3) Move to the parent folder of the main project folder
4) Create a virtualenv
5) Activate the virtualenv
6) Move into the project main folder
7) Install other dependencies by running: pip install -r requirements.txt
8) Place the CSVs into data/csvs/


# How to run
1) To create report analysis from a CSV, run:

python analyze.py <set_csv_filepath>

Reports are saved to data/reports/


2) To experiment some model, run:

python experiment.py <train_set_csv_filepath> <model_initials>

Possible <model_initials> are:
"lg" for Logistic Regression
"sgd" for Stochasting Gradient Descent Regression

Models are saved to data/models/


3) To make predictions on a CSV, run:

python predict.py <test_set_csv_filepath> <model_filepath> <prediction_csv_filename>

Predictions are saved to data/csvs/


4) To generate the best predictions.csv, run:

python experiment.py <train_set_csv_filepath> "lg"
python predict.py <test_set_csv_filepath> data/ predictions.csv