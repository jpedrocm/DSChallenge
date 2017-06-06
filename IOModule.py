##############################################################################
import pandas as pd

from sklearn.externals import joblib


class IOProcessor:
    """Utils class to process IO in all other scripts."""

    @staticmethod
    def _generate_filepath(path, filename, extension, is_train):
        """Returns a full filename along with its path, extension, and
        distintion by dataset.
        """

        if is_train:
            dataset = 'train'
        else:
            dataset = 'test'

        return "".join([path, filename, '_', dataset, extension])

    @classmethod
    def _stringnize_header_report(cls, report):
        """Convert the header report to a formatted string."""

        delim = "######################################"
        
        header = cls._stringnize_pair('Header', report['header'])
        exp_type = cls._stringnize_pair('Expected type', 
                                        report['expected_type'])
        type_stats = cls._stringnize_pair('Type stats', report['type_stats'])
        val_stats = cls._stringnize_pair('Value stats', report['value_stats'])
        
        final_str = "%s\n%s%s%s%s\n%s" %(delim, header, exp_type, type_stats,
                                         val_stats, delim)
        
        return final_str

    @staticmethod
    def _stringnize_pair(header, content):
        """Convert a header-content from the report to a formatted string."""

        return "\n%s%s%s\n" % (header, ': ', str(content))

    @staticmethod
    def load_model(filepath):
        """Loads and returns the ML model from the file in filepath."""

        return joblib.load(filepath)

    @classmethod
    def read_dataset(cls, filepath):
        """Returns a dataframe read from the CSV in the given filepath."""

        df = pd.read_csv(filepath, infer_datetime_format=True)
        return df

    @staticmethod
    def store_model(model, filepath):
        """Stores the ML model in the given filepath."""

        joblib.dump(model, filepath)

    @classmethod
    def write_report(cls, filepath, filename, report, is_train):
        """Writes a general dataset report to a .txt file."""

        full_filepath = cls._generate_filepath(filepath, filename,'.report',
                                               is_train)
        print full_filepath
        cls._write_to_file(full_filepath, report)

    @classmethod
    def write_header_reports(cls, filepath, reports, is_train):
        """Writes header reports to .txt files."""

        for report in reports:
            report_str = cls._stringnize_header_report(report)
            cls.write_report(filepath, report['header'],
                                              report_str, is_train)
 
    @classmethod
    def write_to_csv(cls, filepath, ids_frame, probs):
        """Writes the given probabilities to a CSV located in filepath."""

        csv_dataframe = ids_frame.to_frame()
        csv_dataframe['predictions'] = probs
        csv_dataframe.to_csv(filepath, columns=['ids', 'predictions'],
                             index=False)

    @staticmethod
    def _write_to_file(filepath, content):
        """Writes the given content to a file located in filepath."""

        with open(filepath , 'w') as f:
            f.write(str(content))