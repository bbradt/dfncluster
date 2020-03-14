import json
from dfncluster.Dataset import NiftiDataset


def make():
    dataset = NiftiDataset(filename="data/NiftiDatasets/TestNiftiDataset/data.csv",
                           feature_columns=['filename'],
                           label_columns=['label'])
    dataset.save('data/NiftiDatasets/TestNiftiDataset/test_nifti_dataset')
    return dataset
