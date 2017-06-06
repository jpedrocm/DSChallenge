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

    @staticmethod
    def adapt_test_features(test_dataframe, TRAIN_HEADERS):
        """Removes, adds and reorders the dataframe based on its headers and
        the headers from the training set, so both frames have the same number
        of features.
        """

        test_headers = np.array(test_dataframe.columns)
        del_test_headers = filter(lambda h: h not in TRAIN_HEADERS, 
                                  test_headers)
        test_dataframe.drop(del_test_headers, inplace=True, axis=1)
        
        add_test_headers = filter(lambda h: h not in test_headers,
                                  TRAIN_HEADERS)
        for header in add_test_headers:
            test_dataframe[header] = np.zeros(test_dataframe.shape[0])

        test_dataframe = test_dataframe[TRAIN_HEADERS]

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

        ids_frame = dataframe[id_header].copy()

        if label_header is None:
            y = None
            dataframe.drop([id_header], axis=1, inplace=True)
        else:
            y = np.array(dataframe[label_header]).astype(bool)
            dataframe.drop([id_header, label_header], axis=1, inplace=True)
          
        X = np.array(dataframe).astype(np.float64)

        del dataframe

        return X, y, ids_frame

    @staticmethod
    def undersample(dataframe):
        """Do undersampling on the dataframe."""

        df_false = dataframe[dataframe.default==False]
        df_true = dataframe[dataframe.default==True]

        under_df_false = df_false.sample(df_true.shape[0]*2)
        new_dataframe = pd.concat([under_df_false, df_true])

        return new_dataframe

    @staticmethod
    def oversample(dataframe):
        """Do undersampling on the dataframe."""

        df_false = dataframe[dataframe.default==False]
        df_true = dataframe[dataframe.default==True]

        new_dataframe = pd.concat([df_false, df_true, df_true])

        return new_dataframe