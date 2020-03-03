import json
from dfncluster.Dataset import NiftiDataset

if __name__=='__main__':
    dataset = NiftiDataset(filename="dfncluster/data/NiftiDatasets/TestNiftiDataset/data.csv",
                           feature_columns=['filename'],
                           label_columns=['label'])
    dataset.save('dfncluster/data/NiftiDatasets/TestNiftiDataset/test_nifti_dataset')