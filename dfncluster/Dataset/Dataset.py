import sklearn.model_selection as skms
import pandas as pd
import numpy as np


class Dataset(object):
    """
        Dataset class. Abstract class for creating data sets.
        Each Dataset has:
                fieldname           |   type            |   description
                --------------------|-------------------|----------------
                features            |   numpy ndarray   |   a numpy array of features
                labels              |   numpy ndarray   |   a numpy array of labels
                idx                 |   numpy ndarray   |   a numpy array of instance indices
                unique_labels       |   numpy ndarray   |   a numpy array of the unique labels
                num_instances       |	int	            |	the number of instances
                num_features        |   N-tuple         |   tuple of features in each dataset dimension (most often a 1-tuple)
                num_labels          |   N-tuple         |   tuple of label space in each label dimension    (most often a 1-tuple)
                num_unique_labels   |   int             |   number of unique labels (across all label dimensions)
                label_indices       |   dict            |   a map of indices which map particular unique labels
        Each Dataset does:
                function name   |   return type         |   args            |   kwargs              |   description
                ----------------|-----------------------|-------------------|-----------------------|--------------------
                __init___       |   Dataset             |   self            |   shuffle, **kwargs   |   class constructor
                split           |   (ndarray) 4-tuple   |   test_size       |   **kwargs            |   split dataset into X_train, y_train, X_test, y_test
                generate        |   (ndarray, ndarray)  |   self            |   **kwargs            |   generates the dataset (features, labels)
                shuffle         |   ndarray             |   self            |   **kwargs            |   shuffle the dataset, returning
                save            |   None                |   self            |   prefix              |   serialize the dataset object in numpy
                load            |   Dataset             |   class, filename |   -                   |   classmethod to load a dataset of given type or sub-type of Dataset
    """

    def __init__(self, shuffle_instances=True, **kwargs):
        """
            Constructor for Dataset super-class. Generally should not be instantiated.
            Usage:
                dataset = Dataset(shuffle=True,k1=v1,k2=v2,**kwargs)
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                shuffle         boolean     |   True        |       Whether or not to shuffle the dataset
                NOTE: Other kwargs are passed to the self.generate function
            Args:
                -
            Return:
                Instantiated Dataset Object
        """
        x, y = self.generate(**kwargs)
        if x.ndim == 1:
            x = x.reshape(np.shape(x)[0], 1)
        if y.ndim == 1:
            y = y.reshape(np.shape(y)[0], 1)
        self.features = x
        self.labels = y
        print("Feature Shape %s\nLabel Shape %s" %
              (str(x.shape), str(y.shape)))
        self.num_instances = np.shape(x)[0]
        self.num_features = np.shape(x)[1:]         # can be an N-Tuple if the dataset is multi-dimensional
        self.num_labels = np.shape(y)[1:]
        self.idx = np.arange(self.num_instances)
        self.unique_labels = np.unique(self.labels)
        self.num_unique_labels = self.unique_labels.size
        if shuffle_instances:
            self.shuffle()
        self.label_indices = {label: self.idx[np.all(self.labels == label, axis=1)]
                              for label in self.unique_labels}

    def split(self, test_size, **kwargs):
        """
            get the split features and labels from a dataset
            Usage:
                dataset = Dataset(shuffle=True,k1=v1,k2=v2,**kwargs)
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                ------------|---------------|---------------|---------------------------------------------
                shuffle     |   boolean     |   True        |       Whether or not to shuffle the dataset
                NOTE: Other kwargs are passed to the sklearn train_test_split function
            Args:
                name        |   type        |   Description
                ------------|---------------|--------------
                test_size   |   float       |   size of the test dataset (between 0, and 1)
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                X_train |   ndarray |   n_train x features x ...    |  the training features
                y_train |   ndarray |   n_train x labels x ...      |  the training labels
                X_test  |   ndarray |   n_test  x features x ...    |  the testing features
                y_test  |   ndarray |   n_test  x labels x ...      |  the testing labels
            End-State:
                -
        """
        X_train, X_test, y_train, y_test = skms.train_test_split(
            self.features, self.labels, test_size=test_size, **kwargs)
        return X_train, y_train, X_test, y_test

    def generate(self, **kwargs):
        """
            Usage:
                dataset = Dataset(shuffle=True,k1=v1,k2=v2,**kwargs)
                dataset.generate()  # redundant since generate is called
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                ------------|---------------|---------------|---------------------------------------------
                shuffle     |   boolean     |   True        |       Whether or not to shuffle the dataset
            Args:
                -
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                x       |   ndarray |   instances x features x ...  |  the dataset features
                y       |   ndarray |   instances x labels x ...    |  the dataset labels
            End-State:
                -
            #TODO: Make this a class method?
        """
        x = np.array([[0], [1]])
        y = np.array([[0], [1]])
        return x, y

    def shuffle(self, **kwargs):
        """
           Usage:
                dataset = Dataset(**kwargs)
                idx = dataset.shuffle()
            Kwargs:
                -
            Args:
                -
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                idx     |   ndarray |   instances                   |  the shuffled instance ids
            End-State:
                idx field is shuffled in place
                features field is shuffled in place according to shuffled idx
                labels field is shuffled in place according to shuffled idx
        """
        np.random.shuffle(self.idx)
        self.features = self.features[self.idx, ...]
        self.labels = self.labels[self.idx, ...]
        return self.idx

    def save(self, prefix="dataset"):
        """
           Usage:
                dataset = Dataset(**kwargs)
                idx = dataset.save(prefix="my_dataset")
            Kwargs:
                -
            Args:
                -
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                prefix  |   str     |   -                           |  the file prefix for saving
            End-State:
                the dataset object prefix.npy is saved
        """
        np.save(prefix, self, allow_pickle=True)

    @classmethod
    def load(cls, filename):
        """
           Usage:
                dataset = Dataset(**kwargs)
                idx = dataset.save(prefix="my_dataset")
                dataset_loaded = Dataset.load("my_dataset.npy")
            Kwargs:
                -
            Args:
                name        |   type        |   Description
                ------------|---------------|--------------
                filename    |   str         |   existing filename to load dataset from
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                loaded  |   Dataset |   -                           |  the dataset loaded according to cls type. So sub-classes will load as their own type.
            End-State:
                -
        """
        loaded = np.load(filename, allow_pickle=True).item()
        loaded.__class__ = cls
        return loaded
