import pandas as pd

from dfncluster.Dataset import Dataset


class CsvDatset(Dataset):
    """Abstract class for reading CSVs"""

    def __init__(self, filename=None, feature_columns=[], label_columns=[]):
        """
             args:
                filename	string	name of the CSV to load
                feature_columns	list<string>	labels of feature columns
                label_columns	list<string>	labels of label columns
        """
        super(CsvDataset, self).__init__(filename=filename,
                                         feature_columns=feature_columns, label_columns=label_columns)

    def generate(self,  **kwargs):
        full_data = pd.read_csv(kwargs['filename'])
        features = full_data[kwargs['feature_columns']]
        labels = full_data[kwargs['label_columns']]
        x = features.to_numpy()
        y = labels.to_numpy()
        return x, y
