##############################################################################
import pandas as pd


class Handler:
	"""This class handles missing data in the dataframe and removes
	unninformative columns from it.
	"""

	@classmethod
    def _identify_imputation_method(cls, method):
        if method=='mean':
        	return cls._impute_mean_value
        elif method=='mode':
        	return cls._impute_mode_value
        elif method=='median':
        	return cls._impute_median_value
        else:
        	return cls._impute_previous_value

	@staticmethod
	def _impute_mean_value(column):
		"""Fill missing data with the mean of the column."""
        
        mean_val = column.mean()
        column.fillna(value=mean_val, inplace=True)

	@staticmethod
	def _impute_median_value(column):
		"""Fill missing data with the median of the column."""

		median_val = column.median()
		column.fillna(value=median_val, inplace=True)

	@staticmethod
	def _impute_mode_value(column):
		"""Fill missing data with the mode of the column."""

		mode_val = column.mode()
		column.fillna(value=mode_val, inplace=True)

	@staticmethod
	def _impute_previous_value(column):
		"""Fill missing data with previous values present in the column."""

		column.fillna(method='ffil', inplace=True)

	@classmethod
    def impute_missing_values(cls, dataframe, headers, method = None):
    	"""Impute data for the missing values in the specified columns with 
    	the given method.
    	"""

    	_impute_function = cls._identify_imputation_method(method)

        for header in headers:
        	column = dataframe[header]
        	_impute_function(column)	
    
    @staticmethod
	def remove_columns(dataframe, headers):
		"""Removes unwanted columns in place based on the given list of 
		headers.
		"""

        dataframe.drop(headers, inplace=True, axis=1)