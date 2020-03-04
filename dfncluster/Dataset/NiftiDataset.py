import nibabel as nib
from dfncluster.Dataset import FileFeatureDataset


def load_wrapper(filename):
    return nib.load(filename).dataobj


class NiftiDataset(FileFeatureDataset):

    def __init__(self, **kwargs):
        super(NiftiDataset, self).__init__(loader=load_wrapper, **kwargs)
