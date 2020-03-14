import json
from dfncluster.Dataset import NiftiDataset


class TestNiftiDataset:
    @staticmethod
    def make():
        dataset = NiftiDataset(filename="data/NiftiDatasets/TestNiftiDataset/data.csv",
                               feature_columns=['filename'],
                               label_columns=['label'])
        dataset.save('data/NiftiDatasets/TestNiftiDataset/test_nifti_dataset')
        return dataset


if __name__ == '__main__':
    dataset = TestNiftiDataset.make()
