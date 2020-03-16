import numpy as np
from dfncluster.Dataset import Dataset


class GaussianDataset(Dataset):
    """
        GaussianDataset class. Extends Dataset superclass.
        Generates gaussian features with parameters per class.
        Each GaussianDataset has:
            All fields from Dataset
        Each GaussianDataset does:
            All methods from Dataset
    """

    def __init__(self, parameters=[dict(sigma=1, mu=-1, N=1024), dict(sigma=1, mu=1, N=1024)], num_features=2):
        """
            Constructor for GaussianDataset.
            Usage:
                dataset = GaussianDataset(shuffle=True,parameters=[dict(sigma=1, mu=-1, N=1024), dict(sigma=1, mu=1, N=1024)], num_features=2,**kwargs)
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                parameters  |   list<dict>  |               |   list of dictionaries describing sigma and mu parameters for each class
                num_features|   int         |   2           |   the number of gaussian features to generate for all classes
                NOTE: Other kwargs are passed to the self.generate function
            Args:
                -
            Return:
                Instantiated GaussianDataset Object
        """
        super(GaussianDataset, self).__init__(
            parameters=parameters, num_features=num_features)

    def generate(self,  **kwargs):
        """
            Usage:
                dataset = GaussianDataset(shuffle=True,parameters=[dict(sigma=1, mu=-1, N=1024), dict(sigma=1, mu=1, N=1024)], num_features=2,**kwargs)
                dataset.generate()  # redundant since generate is called
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                parameters  |   list<dict>  |               |   list of dictionaries describing sigma and mu parameters for each class
                num_features|   int         |   2           |   the number of gaussian features to generate for all classes
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
