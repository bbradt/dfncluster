import pandas as pd
import numpy as np
from dfncluster.Dataset import CsvDataset
from itertools import zip_longest


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

    def ragged_pad_stack(self, arrays):
        ndims = arrays[0].ndim

    def generate(self,  **kwargs):
        loader = kwargs['loader']
        features, labels = super(FileFeatureDataset, self).generate(**kwargs)
        x = []
        for instance in features:
            try:
                x.append(loader(instance[0])[np.newaxis, ...])
            except TypeError:
                continue
            except FileExistsError:
                continue
        return np.squeeze(np.stack(zip_longest(*x, fillvalue=0), 0)), labels
