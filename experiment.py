##############################################################################
import sys

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation


HEADERS_REMOVALBLE = ['ids']
HEADERS_MEAN = []
HEADERS_MEDIAN = []
HEADERS_MODE = []
HEADERS_PREVIOUS = ['zip']


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

    Handler.remove_columns(df, HEADERS_REMOVALBLE)
    Handler.impute_missing_values(df, HEADERS_MEAN, 'mean')
    Handler.impute_missing_values(df, HEADERS_MEDIAN, 'median')
    Handler.impute_missing_values(df, HEADERS_MODE, 'mode')
    Handler.impute_missing_values(df, HEADERS_PREVIOUS)

