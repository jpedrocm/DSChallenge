##############################################################################
import sys

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation

REMOVABLE_HEADERS = ['']

if __name__=='__main__':
    """This script generates a ML model trained on the training set."""

    try:
        train_filepath = sys.argv[1]        
        model_filepath = sys.argv[2]
        model_initials = sys.argv[3]
    except:
        raise Exception("Missing one or multiple sys args.")

    
    df = IOProcessor.read_dataset(train_filepath)

    Handle.remove_columns(df, )