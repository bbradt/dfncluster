import pandas as pd
import numpy as np
from dfncluster.Dataset import CsvDataset


class FileFeatureDataset(CsvDataset):
    def __init__(self, loader=None,  feature_columns=['filename'], label_columns=['label'], **kwargs):
        """
             args:
                filename	string	name of the CSV to load
                feature_columns	list<string>	labels of feature columns
                label_columns	list<string>	labels of label columns
        """
        super(FileFeatureDataset, self).__init__(loader=loader,
                                                 feature_columns=feature_columns,
                                                 label_columns=label_columns,
                                                 **kwargs)

    def generate(self,  **kwargs):
        loader = kwargs['loader']
        features, labels = super(FileFeatureDataset, self).generate(**kwargs)
        x = []
        for instance in features:
            x.append(loader(instance[0])[np.newaxis, ...])
        return np.squeeze(np.stack(x), 0), labels
