import numpy as np
from dfncluster.Dataset import SklearnDataset

KWARGS = dict(
    n_samples=10000,
    n_features=100,
    centers=None,
    cluster_std=1.0,
    center_bos=(-10.0, 10.0),
    shuffle=False,
    random_state=314159
)

if __name__ == '__main__':
    dataset = SklearnDataset(dataset_name='make_blobs', **KWARGS)
    dataset.save('data/SklearnDatasets/Blobs/blobs')
