import pandas as pd
import numpy as np
from dfncluster.Dataset import CsvDataset


class FileFeatureDataset(CsvDataset):
    """
        FileFeatureDataset class. Extends the CsvDataset class. See the docstring for Dataset for inherited fields/functions.
        Generates a dataset based on a CSV or TSV file, with the added ability of using a loader function to load feature_columns specified by files.
        Each FileFeatureDataset has:
            All inherited fields from CsvDataset
        Each Dataset does:
            All inherited functions from CsvDataset
            ---
            function name   |   return type         |   args            |   kwargs              |   description
            ----------------|-----------------------|-------------------|-----------------------|--------------------
            __init___       |   CsvDataset            |   self            |   shuffle, **kwargs   |   subclass constructor
            generate        |   (ndarray, ndarray)  |   self            |   **kwargs            |   overrides superclass function, generates the dataset (features, labels)
    """

    def __init__(self, loader=None,  feature_columns=['filename'], label_columns=['label'], **kwargs):
        """
            Constructor for FileFeatureDataset.
            Usage:
                dataset = FileFeatureDataset(shuffle=True,loader=np.imread,feature_columns=['feature_1','feature_2'],label_columns=['label'],**kwargs)
            Kwargs:
                keyword         |   type        |   default     |       Description                                    
                ----------------|---------------|---------------|-------------------
                filename        |   str         |   None        |   filename to load dataset from
                loader          |   function    |   None        |   function handle for feature loading function
                feature_columns |   list<str>   |   ['feature'] |   list of column names for features
                label_columns   |   list<str>   |   ['label']   |   list of column names for labels
                NOTE: Other kwargs are passed to the CsvDataset superclass constructor (and thus to generate, and shuffle methods)
            Args:
                -
            Return:
                Instantiated FileFeatureDataset Object
        """
        super(FileFeatureDataset, self).__init__(loader=loader,
                                                 feature_columns=feature_columns,
                                                 label_columns=label_columns,
                                                 **kwargs)

    def generate(self,  **kwargs):
        """
            Usage:
                dataset = FileFeatureDataset(shuffle=True,loader=np.imread,feature_columns=['feature_1','feature_2'],label_columns=['label'],**kwargs)
                dataset.generate()
            Kwargs:
                NOTE: These kwargs are pulled from the kwargs dictionary directly, since they are passed from the constructor.
                keyword         |   type        |   default     |       Description                                    
                ----------------|---------------|---------------|-------------------
                filename        |   str         |   None        |   filename to load dataset from
                feature_columns |   list<str>   |   ['feature'] |   list of column names for features
                label_columns   |   list<str>   |   ['label']   |   list of column names for labels
            Args:
                -
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                x       |   ndarray |   instances x features x ...  |  the dataset features
                y       |   ndarray |   instances x labels x ...    |  the dataset labels
            End-State:
                -
        """
        loader = kwargs['loader']
        features, labels = super(FileFeatureDataset, self).generate(**kwargs)
        x = []
        for instance in features:
            x.append(loader(instance[0])[np.newaxis, ...])
        return np.squeeze(np.stack(x, 0)), labels
