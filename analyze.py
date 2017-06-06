##############################################################################
import sys

import numpy as np

from IOModule import IOProcessor
from AnalyzerModule import Analyzer


REPORT_FILEPATH = 'reports/'

expected_types = {'ids': str,
                  'default': bool,
                  'score_1':str, 
                  'score_2':str, 
                  'score_3': np.float64,
                  'score_4': np.float64,
                  'score_5': np.float64,
                  'score_6': np.float64,
                  'risk_rate': np.float64,
                  'amount_borrowed': np.float64,
                  'borrowed_in_months': np.float64,
                  'credit_limit': np.float64,
                  'reason': str,
                  'income': np.float64,
                  'sign': str,
                  'gender': str,
                  'facebook_profile': bool,
                  'last_payment': str, 
                  'end_last_loan': str,                      
                  'state': str,
                  'zip': str, 
                  'channel': str,
                  'job_name': str,
                  'real_state': str,
                  'ok_since': np.float64,
                  'n_bankruptcies': np.float64,
                  'n_defaulted_loans': np.float64,
                  'n_accounts': np.float64,
                  'n_issues': np.float64}


if __name__=='__main__':
    """This script generates reports from the given csv."""

    try:
        csv_filepath = sys.argv[1]
        is_train = 'train' in csv_filepath
    except:
        raise Exception("Missing csv path.")

    print "Reading dataset"
    df = IOProcessor.read_dataset(csv_filepath)

    print "Writing column reports"
    header_reports = Analyzer.create_header_reports(df, expected_types)
    IOProcessor.write_header_reports(REPORT_FILEPATH, header_reports, is_train)

    print "Writing general report"
    general_report = Analyzer.create_general_report(df)
    IOProcessor.write_report(REPORT_FILEPATH, 'general_report', general_report, is_train)