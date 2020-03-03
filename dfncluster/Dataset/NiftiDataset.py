import nibabel as nib
from dfncluster.Dataset import FileFeatureDataset


class NiftiDataset(FileFeatureDataset):

    def __init__(self, **kwargs):
        super(NiftiDataset, self).__init__(loader=nib.load, **kwargs)
