import numpy as np
from dfncluster.Dataset import Dataset


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
