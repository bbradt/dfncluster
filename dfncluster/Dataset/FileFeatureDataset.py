import pandas as pd
import numpy as np
from dfncluster.Dataset import CsvDataset


class FileFeatureDataset(CsvDataset):
    def __init__(self, loader=None, **kwargs):
        """
             args:
                filename	string	name of the CSV to load
                feature_columns	list<string>	labels of feature columns
                label_columns	list<string>	labels of label columns
        """
        super(FileFeatureDataset, self).__init__(loader=loader, **kwargs)

    def generate(self,  **kwargs):
        loader = kwargs['loader']
        features, labels = super(FileFeatureDataset, self).generate(**kwargs)
        x = []
        for instance in features:
            x.append([loader(col) for col in instance])
        return np.array(x), labels
