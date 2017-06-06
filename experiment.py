##############################################################################
import sys

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation

from VariablesModule import N_FOLDS, MODEL_DICT
from VariablesModule import ROWS_REMOVABLES_ALL, ROWS_REMOVABLES_ANY
from VariablesModule import HEADERS_REMOVALBLE, HEADERS_MEAN, HEADERS_MODE
from VariablesModule import HEADERS_MEDIAN, HEADERS_PREVIOUS
from VariablesModule import HEADERS_BOOLEAN, HEADERS_CATEGORICAL


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

    try:
        model = MODEL_DICT[model_initials]
    except:
        raise Exception("Model not found in MODEL DICT.")

    
    df = IOProcessor.read_dataset(train_filepath)

    Handler.remove_rows(df, ROWS_REMOVABLES_ALL, 'all')
    Handler.remove_rows(df, ROWS_REMOVABLES_ANY, 'any')
    Handler.remove_columns(df, HEADERS_REMOVALBLE)

    Handler.impute_missing_values(df, HEADERS_MEAN, 'mean')
    Handler.impute_missing_values(df, HEADERS_MEDIAN, 'median')
    Handler.impute_missing_values(df, HEADERS_MODE, 'mode')
    Handler.impute_missing_values(df, HEADERS_PREVIOUS)

    Encoder.encode_booleans(df, HEADERS_BOOLEAN)
    encoded_df = Encoder.encode_categoricals(df, HEADERS_CATEGORICAL)
    print encoded_df.columns[:50]
    raise NameError
    X, y, ids_frame = Encoder.divide_dataframe(encoded_df, 'default', 'ids')

    exp = Experimentation(model, N_FOLDS)
    f1_folds, avg, dev = exp.experiment_model(X, y)

    print "F1 scores: " + str(f1_folds)
    print "Averaged F1: " + str(avg)
    print "Deviation F1: " + str(dev)

    trained_model = exp.get_model()
    IOProcessor.store_model(trained_model, model_filepath)