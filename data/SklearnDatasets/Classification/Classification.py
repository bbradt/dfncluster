import numpy as np
from dfncluster.Dataset import SklearnDataset

KWARGS = dict(
    n_samples=10000,
    n_features=100,
    n_informative=47,
    n_redundant=53,
    n_repeated=0,
    n_classes=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=False,
    random_state=123
)


class Classification:
    @staticmethod
    def make():
        dataset = SklearnDataset(dataset_name='make_classification', **KWARGS)
        dataset.save('data/SklearnDatasets/Classification/classification')
        return dataset


if __name__ == '__main__':
    Classification.make()
