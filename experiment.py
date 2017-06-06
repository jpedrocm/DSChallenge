##############################################################################
import sys

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation

from VariablesModule import N_FOLDS, MODEL_DICT
from VariablesModule import ROWS_REMOVABLE_ALL, ROWS_REMOVABLE_ANY
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

    print "Reading training set"
    df = IOProcessor.read_dataset(train_filepath)

    print "Removing unninformative rows and columns"
    Handler.remove_rows(df, ROWS_REMOVABLE_ALL, 'all')
    Handler.remove_rows(df, ROWS_REMOVABLE_ANY, 'any')
    Handler.remove_columns(df, HEADERS_REMOVALBLE)

    print "Imputing missing data"
    Handler.impute_missing_values(df, HEADERS_MEAN, 'mean')
    Handler.impute_missing_values(df, HEADERS_MEDIAN, 'median')
    HEADERS_MODE.append('default')
    Handler.impute_missing_values(df, HEADERS_MODE, 'mode')
    Handler.impute_missing_values(df, HEADERS_PREVIOUS)

    print "Encoding features and labels"
    Encoder.encode_booleans(df, HEADERS_BOOLEAN)
    encoded_df = Encoder.encode_categoricals(df, HEADERS_CATEGORICAL)
    X, y, ids_frame = Encoder.transform_and_del_dataframe(encoded_df, 
                                                         'default', 'ids')
    print "(rows, features) = " + str(X.shape)

    print "Evaluating model"
    exp = Experimentation(model, N_FOLDS)
    f1_folds, avg, dev = exp.experiment_model(X, y)

    print "\n################################"
    print "F1 scores: " + str(f1_folds)
    print "F1 mean: " + str(avg)
    print "F1 deviation: " + str(dev)
    print "################################\n"

    print "Training model"
    exp.train_model(X, y)
    print exp.predict_probs(X)
    print "Storing model"
    trained_model = exp.get_model()
    IOProcessor.store_model(trained_model, model_filepath)

    print "Done"