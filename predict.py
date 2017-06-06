##############################################################################
import sys

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation

from VariablesModule import N_FOLDS, MODEL_DICT, HEADERS_TRAIN_FILENAME
from VariablesModule import HEADERS_REMOVALBLE, HEADERS_MEAN, HEADERS_MODE
from VariablesModule import HEADERS_MEDIAN, HEADERS_PREVIOUS
from VariablesModule import HEADERS_BOOLEAN, HEADERS_CATEGORICAL



if __name__=='__main__':
    """This script generates predictions probabilities based on the given 
    trained model.
    """

    try:
        test_filepath = sys.argv[1]        
        model_filepath = sys.argv[2]
        predictions_filepath = sys.argv[3]
    except:
        raise Exception("Missing one or multiple sys args.")
    
    try:
        model = IOProcessor.load_model(model_filepath)
        headers_train = IOProcessor.load_encoded_headers(HEADERS_TRAIN_FILENAME)
    except:
        raise Exception("Model not found in the given path.")

    
    print "Reading test set"
    df = IOProcessor.read_dataset(test_filepath)

    print "Removing unninformative columns"
    Handler.remove_columns(df, HEADERS_REMOVALBLE)

    print "Imputing missing data"
    Handler.impute_missing_values(df, HEADERS_MEAN, 'mean')
    Handler.impute_missing_values(df, HEADERS_MEDIAN, 'median')
    Handler.impute_missing_values(df, HEADERS_MODE, 'mode')
    Handler.impute_missing_values(df, HEADERS_PREVIOUS)

    print "Encoding features and labels"
    Encoder.encode_booleans(df, HEADERS_BOOLEAN)
    encoded_df = Encoder.encode_categoricals(df, HEADERS_CATEGORICAL)
    print encoded_df.shape
    Encoder.adapt_test_features(encoded_df, headers_train)
    print encoded_df.shape
    X, _ , ids_frame = Encoder.transform_and_del_dataframe(encoded_df, 
                                                         None, 'ids')
    print "(rows, features) = " + str(X.shape)

    raise NameError

    print "Predicting probabilities"
    exp = Experimentation(model, N_FOLDS)
    probs = exp.predict_probs(X)

    print probs

    raise NameError

    print "Writing results"
    IOProcessor.write_to_csv(predictions_filepath, ids_frame, probs)

    print "Done"