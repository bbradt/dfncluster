"""
    MatDataset
        Loads files stored in .mat format.
        Stacks variables in .mat files
"""
import numpy as np
import scipy.io as sio
from dfncluster.Dataset import FileFeatureDataset


def load_wrapper(filename):
    loaded = sio.loadmat(filename)
    return np.stack([loaded[k] for k in loaded.keys() if '__' not in k])


class MatDataset(FileFeatureDataset):

    def __init__(self, **kwargs):
        """Uses the FileFeature Dataset to load features/labels from a CSV.
        Note that following the FileFeature protocol, only features are loaded from the parent.
        """
        super(MatDataset, self).__init__(loader=load_wrapper, **kwargs)
