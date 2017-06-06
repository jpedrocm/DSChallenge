##############################################################################

import pandas as pd
import numpy as np

class Encoder:
    """This class encodes boolean and categorical features to types the model
    can receive as input. Also encodes the labels.
    """

    @staticmethod
    def _encode_boolean(column):
        """Do in place boolean2int encoding in the column."""

        return column.replace(to_replace=[True, False], value=[1, 0],
                                          inplace=True)

    @classmethod
    def encode_booleans(cls, dataframe, headers):
        """Do boolean encoding in each header column from headers list."""

        for header in headers:
            column = dataframe[header]
            cls._encode_boolean(column)

    @staticmethod
    def encode_categoricals(dataframe, headers):
        """Do one hot encoding for columns with a header in headers.
        Not in place.
        """

        return pd.get_dummies(dataframe, columns = headers)

    @staticmethod
    def transform_and_del_dataframe(dataframe, label_header, id_header):
        """Transform the given dataframe into three parts and returns them.
        The second is the column frame of labels, the third is the column 
        frame of IDs and the first are all the remaining columns.
        """

        if label_header is None:
            y = None
        else:
            y = np.array(dataframe[label_header]).astype(bool)

        ids = np.array(dataframe[id_header])

        dataframe.drop([id_header, label_header], axis=1, inplace=True)          
        X = np.array(dataframe).astype(np.float64)

        del dataframe

        return X, y, ids