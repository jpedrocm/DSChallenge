##############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier


"""This module contains all variables for experimentation and prediction."""


N_FOLDS = 5

ROWS_REMOVABLES_ALL = ['n_accounts', 'n_issues', 'n_bankruptcies', 'ok_since',
                   'n_defaulted_loans', 'real_state', 'job_name']

ROWS_REMOVABLES_ANY = ['state', 'zip', 'last_payment', 'end_last_loan']

HEADERS_REMOVALBLE = ['ids', 'channel', 'sign', 'ok_since', 'job_name']

HEADERS_MEAN = ['risk_rate', 'income', 'score_3', 'score_4', 'score_5',
                'score_6']

HEADERS_MEDIAN = ['credit_limit', 'amount_borrowed']

HEADERS_MODE = ['default', 'n_bankruptcies', 'n_defaulted_loans', 
                'n_accounts', 'n_issues', 'real_state', 'borrowed_in_months',
                'score_1', 'score_2']

HEADERS_PREVIOUS = ['facebook_profile', 'gender', 'reason', 'state', 'zip']
    #last paymnt, end last loan

MODEL_DICT = {"lg": LogisticRegression(n_jobs = -1, random_state = 14128), 
              "sgd": SGDClassifier(n_jobs = -1, random_state = 14128),
              "rf": RandomForestClassifier(n_jobs = -1, random_state = 14128)}