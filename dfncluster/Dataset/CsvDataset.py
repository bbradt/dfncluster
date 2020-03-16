import pandas as pd

from dfncluster.Dataset import Dataset


class CsvDataset(Dataset):
    """
        CsvDataset class. Extends the Dataset class. See the docstring for Dataset for inherited fields/functions.
        Generates a dataset based on a CSV or TSV file.
        Each CsvDataset has:
            All inherited fields from Dataset
        Each Dataset does:
            All inherited functions from Dataset
            ---
            function name   |   return type         |   args            |   kwargs              |   description
            ----------------|-----------------------|-------------------|-----------------------|--------------------
            __init___       |   CsvDataset            |   self            |   shuffle, **kwargs   |   subclass constructor
            generate        |   (ndarray, ndarray)  |   self            |   **kwargs            |   overrides superclass function, generates the dataset (features, labels)
    """

    def __init__(self, filename=None, feature_columns=['feature'], label_columns=['label'], **kwargs):
        """
            Constructor for CsvDataset. Generally should not be instantiated.
            Usage:
                dataset = CsvDataset(shuffle=True,feature_columns=['feature_1','feature_2'],label_columns=['label'],**kwargs)
            Kwargs:
                keyword         |   type        |   default     |       Description                                    
                ----------------|---------------|---------------|-------------------
                filename        |   str         |   None        |   filename to load dataset from
                feature_columns |   list<str>   |   ['feature'] |   list of column names for features
                label_columns   |   list<str>   |   ['label']   |   list of column names for labels
                NOTE: Other kwargs are passed to the Dataset superclass constructor (and thus to generate, and shuffle methods)
            Args:
                -
            Return:
                Instantiated CsvDataset Object
        """
        super(CsvDataset, self).__init__(filename=filename,
                                         feature_columns=feature_columns,
                                         label_columns=label_columns,
                                         **kwargs)

    def generate(self,  **kwargs):
        """
            Usage:
                dataset = CsvDataset(shuffle=True,feature_columns=['feature_1','feature_2'],label_columns=['label'],**kwargs)
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
        full_data = pd.read_csv(kwargs['filename'])
        features = full_data[kwargs['feature_columns']]
        labels = full_data[kwargs['label_columns']]
        x = features.to_numpy()
        y = labels.to_numpy().flatten()
        return x, y
