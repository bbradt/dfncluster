import json
from dfncluster.Dataset import GaussianConnectivityDataset


class TestGCDataset:
    @staticmethod
    def make():
        dataset = GaussianConnectivityDataset()
        dataset.save('data/GaussianConnectivityDatasets/TestGCDataset/test_gc')
        return dataset


if __name__ == '__main__':
    dataset = TestGCDataset.make()
