##############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier


"""This module contains all variables for experimentation and prediction."""


# NUMBER OF FOLDS IN CROSS VALIDATION
N_FOLDS = 5

# DICT OF (MODEL_INITIAL: MODEL OBJECT)
MODEL_DICT = {"lg": LogisticRegression(n_jobs = -1, random_state = 14128),
              "lg_c": LogisticRegression(class_weight='balanced', n_jobs = -1,
                                         random_state = 14128),
              "sgd": SGDClassifier(n_jobs = -1, random_state = 14128),
              "sgd_log": SGDClassifier(loss='log', class_weight='balanced',
                                       n_jobs = -1, random_state = 14128),
              "rf": RandomForestClassifier(n_jobs = -1, random_state = 14128),
              "rf_50": RandomForestClassifier(n_estimators = 50, n_jobs = -1,
                                              random_state = 14128)}

# ROWS CONTAINING MISSING VALUES IN ALL OF THESE COLUMNS WILL BE REMOVED
ROWS_REMOVABLE_ALL = ['n_accounts', 'n_issues', 'n_bankruptcies', 'ok_since',
                   'n_defaulted_loans', 'real_state', 'job_name']

# ROWS CONTAINING MISSING VALUES IN ANY OF THESE COLUMNS WILL BE REMOVED
ROWS_REMOVABLE_ANY = ['state', 'zip', 'last_payment', 'end_last_loan']

# COLUMNS TO BE REMOVED
HEADERS_REMOVALBLE = ['channel', 'sign', 'ok_since', 'job_name', 'reason',
                      'last_payment', 'end_last_loan']

# COLUMNS TO IMPUTE MISSING VALUES WITH MEAN
HEADERS_MEAN = ['risk_rate', 'income', 'score_3', 'score_4', 'score_5',
                'score_6']

# COLUMNS TO IMPUTE MISSING VALUES WITH MEDIAN
HEADERS_MEDIAN = ['credit_limit', 'amount_borrowed']

# COLUMNS TO IMPUTE MISSING VALUES WITH MODE
HEADERS_MODE = ['default', 'n_bankruptcies', 'n_defaulted_loans', 
                'n_accounts', 'n_issues', 'real_state', 'borrowed_in_months',
                'score_1', 'score_2']

# COLUMNS TO IMPUTE MISSING VALUES WITH SOME VALUE IN THE DISTRIBUTION
HEADERS_PREVIOUS = ['facebook_profile', 'gender', 'state', 'zip']
    #last paymnt, end last loan

# COLUMNS TO ENCODE FROM BOOLEANS TO 1's AND 0's
HEADERS_BOOLEAN = ['facebook_profile']

# COLUMNS TO ONE-HOT ENCODE
HEADERS_CATEGORICAL = ['gender', 'real_state', 'score_1', 'score_2', 
                       'state','zip']