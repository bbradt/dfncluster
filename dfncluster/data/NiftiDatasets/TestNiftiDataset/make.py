import json
from dfncluster.Dataset import NiftiDataset

if __name__=='__main__':
    dataset = NiftiDataset(filename="data/NiftiDatasets/TestNiftiDataset/data.csv")
    dataset.save('data/NiftiDatasets/TestNiftiDataset/test_nifti_dataset')