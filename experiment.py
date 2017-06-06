##############################################################################
import sys

##############################################################################

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation


ROWS_REMOVABLES_1 = ['n_accounts', 'n_issues', 'n_bankruptcies', 'ok_since',
                   'n_defaulted_loans', 'real_state', 'job_name']

ROWS_REMOVABLES_2 = ['state', 'zip']

ROWS_REMOVABLES_3 = ['last_payment', 'end_last_loan']

HEADERS_REMOVALBLE = ['ids', 'channel', 'sign', 'ok_since', 'job_name']

HEADERS_MEAN = ['risk_rate', 'income', 'score_3', 'score_4', 'score_5',
                'score_6']

HEADERS_MEDIAN = ['credit_limit', 'amount_borrowed']

HEADERS_MODE = ['default', 'n_bankruptcies', 'n_defaulted_loans', 
                'n_accounts', 'n_issues', 'real_state', 'borrowed_in_months',
                'score_1', 'score_2']

HEADERS_PREVIOUS = ['facebook_profile', 'gender', 'reason', 'state', 'zip']
    #last paymnt, end last loan

if __name__=='__main__':
    """This script evaluates and generates ML model trained on the 
    training set.
    """

    try:
        train_filepath = sys.argv[1]        
        model_filepath = sys.argv[2]
        model_initials = sys.argv[3]
    except:
        raise Exception("Missing one or multiple sys args.")

    
    df = IOProcessor.read_dataset(train_filepath)

    Handler.remove_rows(df, ROWS_REMOVABLES_1)
    Handler.remove_rows(df, ROWS_REMOVABLES_2)
    Handler.remove_rows(df, ROWS_REMOVABLES_3)
    Handler.remove_columns(df, HEADERS_REMOVALBLE)

    Handler.impute_missing_values(df, HEADERS_MEAN, 'mean')
    Handler.impute_missing_values(df, HEADERS_MEDIAN, 'median')
    Handler.impute_missing_values(df, HEADERS_MODE, 'mode')
    Handler.impute_missing_values(df, HEADERS_PREVIOUS)