import numpy as np
from dfncluster.Dataset import SklearnDataset

KWARGS = dict(
)


class Iris():
    @staticmethod
    def make():
        dataset = SklearnDataset(dataset_name='load_iris', **KWARGS)
        dataset.save('data/SklearnDatasets/Iris/iris')
        return dataset


if __name__ == '__main__':
    Iris.make()
