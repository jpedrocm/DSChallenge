##############################################################################
import sys

from IOModule import IOProcessor
from HandlerModule import Handler
from EncoderModule import Encoder
from ExperimentationModule import Experimentation

from VariableModule import N_FOLDS, MODEL_DICT
from VariableModule import ROWS_REMOVABLES_ALL, ROWS_REMOVABLES_ANY
from VariableModule import HEADERS_REMOVALBLE, HEADERS_MEAN, HEADERS_MODE
from VariableModule import HEADERS_MEDIAN, HEADERS_PREVIOUS


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
    except:
        raise Exception("Model not found in the given path.")

    
    df = IOProcessor.read_dataset(test_filepath)

    Handler.remove_rows(df, ROWS_REMOVABLES_ALL, 'all')
    Handler.remove_rows(df, ROWS_REMOVABLES_ANY, 'any')
    Handler.remove_columns(df, HEADERS_REMOVALBLE)

    Handler.impute_missing_values(df, HEADERS_MEAN, 'mean')
    Handler.impute_missing_values(df, HEADERS_MEDIAN, 'median')
    Handler.impute_missing_values(df, HEADERS_MODE, 'mode')
    Handler.impute_missing_values(df, HEADERS_PREVIOUS)

    exp = Experimentation(model, N_FOLDS)
    probs = exp.predict_probs(X)

    IOProcessor.write_to_csv(predictions_filepath, df, probs)