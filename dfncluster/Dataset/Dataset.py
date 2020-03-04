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
        print("Feature Shape %s\nLabel Shape %s" %
              (str(x.shape), str(y.shape)))
        self.num_instances = np.shape(x)[0]
        self.idx = np.arange(self.num_instances)
        self.unique_labels = np.unique(self.labels)
        if shuffle:
            self.shuffle()
        self.label_indices = {
            label: self.idx[self.labels == label] for label in self.unique_labels}

    def split(self, test_size, **kwargs):
        """wraps sklearn.train_test_split"""
        X_train, X_test, y_train, y_test = skms.train_test_split(
            self.features, self.labels, test_size=test_size, **kwargs)
        return X_train, y_train, X_test, y_test

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
    
    def save(self, prefix="dataset"):
        np.save(prefix, self, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """Load a dataset and typecast to own class
        """
        loaded = np.load(filename, allow_pickle=True).item()
        loaded.__class__ = cls
        return loaded