import sklearn.model_selection as skms
import pandas as pd
import numpy as np


class Dataset(object):
    """
        Dataset class. Abstract class for creating data sets.
        Each Dataset has:
                features	numpy ndarray	a numpy array of features
                labels		numpy ndarray	a numpy array of labels
                unique_labels	numpy ndarray	a numpy array of the unique labels
                num_labels	int		number of unique labels
        Each Dataset does:
                generate	x, y		generate a set of features, labels for itself

    """

    def __init__(self, shuffle=True, **kwargs):
        x, y = self.generate(**kwargs)
        self.features = x
        self.labels = y
        self.num_instances = x.shape[0]
        self.idx = np.arange(self.num_instances)
        self.unique_labels = np.unique(self.labels)
        if shuffle:
            self.shuffle()
        self.label_indices = {
            label: self.idx[self.labels == label] for label in self.unique_labels}

    def split(self, test_size, **kwargs):
        """wraps sklearn.train_test_split"""
        X_train, X_test, y_train, y_test = skms.train_test_split(
            self.x, self.y, test_size=test_size, **kwargs)
        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    def generate(self, **kwargs):
        """abstract static method overridden by subclasses"""
        x = np.array([])
        y = np.array([])
        return x, y

    def shuffle(self, **kwargs):
        np.random.shuffle(self.idx)
        self.features = self.features[self.idx, ...]
        self.labels = self.labels[self.idx, ...]
        return self.idx


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


class GaussianDataset(Dataset):
    def __init__(self, parameters=[dict(sigma=1, mu=-1, N=1024), dict(sigma=1, mu=1, N=1024)], num_features=2):
        """
             args:
                filename	string	name of the CSV to load
        """
        super(GaussianDataset, self).__init__(
            parameters=parameters, num_features=num_features)

    def generate(self,  **kwargs):
        parameters = kwargs['parameters']
        num_features = kwargs['num_features']
        features = None
        labels = None
        for label, parameter_set in enumerate(parameters):
            N = parameter_set['N']
            i_features = np.random.normal(
                size=(N, num_features), scale=parameter_set['sigma'], loc=parameter_set['mu'])
            i_labels = np.ones((N, 1)) * label
            if features is None or labels is None:
                features = i_features
                labels = i_labels
            else:
                features = np.vstack((features, i_features))
                labels = np.vstack((labels, i_labels))
        print(labels.shape, features.shape)
        return features, labels.flatten()
