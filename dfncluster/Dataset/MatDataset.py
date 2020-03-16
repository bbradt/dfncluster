import numpy as np
import scipy.io as sio
from dfncluster.Dataset import FileFeatureDataset


def load_wrapper(filename):
    """Wraps the scipy.io mat loader function. Extending functionality, by automatically stacking separate variables.
    TODO: Think about this functionality. Better to return new columns for different variables?
    """
    loaded = sio.loadmat(filename)
    return np.stack([loaded[k] for k in loaded.keys() if '__' not in k])


class MatDataset(FileFeatureDataset):
    """
        MatDataset class. Extends the FileFeatureDataset class. See the docstring for Dataset for inherited fields/functions.
        Generates a dataset based on a CSV or TSV file, where features are given as .mat files.
        Each FileFeatureDataset has:
            All inherited fields from FileFeatureDataset
        Each Dataset does:
            All inherited functions from FileFeatureDataset
    """

    def __init__(self, **kwargs):
        """
            Constructor for MatDataset.
            Usage:
                dataset = MatDataset(shuffle=True,feature_columns=['feature_1','feature_2'],label_columns=['label'],**kwargs)
            Kwargs:
                See documnetation for FileFeatureDataset
            Args:
                -
            Return:
                Instantiated MatDataset Object
        """

        super(MatDataset, self).__init__(loader=load_wrapper, **kwargs)
