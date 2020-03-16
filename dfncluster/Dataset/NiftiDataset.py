import nibabel as nib
from dfncluster.Dataset import FileFeatureDataset


def load_wrapper(filename):
    """wrap the nibabel loader, returning data directly
    """
    return nib.load(filename).dataobj


class NiftiDataset(FileFeatureDataset):
    """
        NiftiDataset class. Extends the FileFeatureDataset class. See the docstring for Dataset for inherited fields/functions.
        Generates a dataset based on a CSV or TSV file, where features are given as .nii or .nii.gz files.
        Each NiftiDataset has:
            All inherited fields from FileFeatureDataset
        Each Dataset does:
            All inherited functions from FileFeatureDataset
    """

    def __init__(self, **kwargs):
        """
            Constructor for NiftiDataset.
            Usage:
                dataset = NiftiDataset(shuffle=True,feature_columns=['feature_1','feature_2'],label_columns=['label'],**kwargs)
            Kwargs:
                See documnetation for FileFeatureDataset
            Args:
                -
            Return:
                Instantiated NiftiDataset Object
        """
        super(NiftiDataset, self).__init__(loader=load_wrapper, **kwargs)
