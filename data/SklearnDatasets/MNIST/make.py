import numpy as np
from dfncluster.Dataset import SklearnDataset

KWARGS = dict(
    name='mnist_784',
    version=1,
    data_id=None,
    data_home='data/SklearnDatasets/MNIST',
    target_column='default-target',
    cache=True,
    as_frame=False
)


if __name__ == '__main__':
    dataset = SklearnDataset(dataset_name='fetch_openml', **KWARGS)
    dataset.save('data/SklearnDatasets/MNIST/mnist')
