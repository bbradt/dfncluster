import numpy as np
from dfncluster.Dataset import SklearnDataset

KWARGS = dict(
)

if __name__ == '__main__':
    dataset = SklearnDataset(dataset_name='load_iris', **KWARGS)
    dataset.save('data/SklearnDatasets/Iris/iris')
