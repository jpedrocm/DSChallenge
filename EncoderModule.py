##############################################################################

import pandas as pd
import numpy as np

class Encoder:
    """This class encodes boolean and categorical features to types the model
    can receive as input. Also encodes the labels."""

    @classmethod
    def encode_booleans(cls, dataframe, headers):
        """Do boolean encoding in each header column from headers list."""

        for header in headers:
            column = dataframe[header]
            cls._encode_boolean(column)

    @staticmethod
    def encode_categoricals(dataframe, headers):
        """Do one hot encoding for columns with a header in headers.
        Not in place."""

        return pd.get_dummies(dataframe, columns = headers)

    @staticmethod
    def _encode_boolean(column):
        """Do in place boolean2int encoding in the column."""

        return column.replace(to_replace=[True, False], value=[1, 0],
                              inplace=True)

    @staticmethod
    def divide_dataframe(dataframe, label_header, id_header):
        """Divide the given dataframe into three parts and returns them. The
        second is the column frame of labels, the third is the column frame of
        IDs and the first are all the remaining columns.
        """

        copied_dataframe = dataframe.drop([id_header, label_header], axis=1)
        X = copied_dataframe.values[:,:-1].astype(np.float64)

        if label_header is None:
            y = None
        else:
            y = dataframe[label_header].values.astype(np.float64)

        ids_frame = dataframe[id_header].copy()

        return X, y, ids_frame