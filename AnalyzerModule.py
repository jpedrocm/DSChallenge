##############################################################################
from pandas import DataFrame
import numpy as np

class Analyzer:
    """This class create reports from the datasets"""
    
    @classmethod
    def _analyze_header(cls, header_frame, header, data_type):
        """Create an analysis report for the given header."""
        
        report = {}
        report['header'] = header
        report['expected_type'] = str(data_type)
        
        type_stats = {}
        value_stats = {}

        for value in header_frame:
            cls._update_stats(type_stats, str(type(value)))
            cls._update_stats(value_stats, value)

        report['type_stats'] = type_stats
        report['value_stats'] = value_stats

        return report

    @staticmethod
    def _update_stats(stats, val):
        """Update the count of the value val in the given stats dict"""

        if val in stats:
            stats[val]+=1
        else:
            stats[val]=1


    @staticmethod
    def create_general_report(dataframe):
        """Returns a general report of all dataframe's numeric headers"""

        return dataframe.describe()
    
    @classmethod
    def create_header_reports(cls, dataframe, hashmap):
        """Create and return reports for each header of the given dataframe
        using the hashmap param. The hashmap is a dict whose keys are strings
        (representing header names) and values are data types.
        """

        headers = dataframe.columns

        analysis = []
        for header, data_type in hashmap.iteritems():
            if header in headers:
                header_frame = dataframe[header]
                header_analysis = cls._analyze_header(header_frame, header, 
                                                      data_type)
                analysis.append(header_analysis)

        return analysis