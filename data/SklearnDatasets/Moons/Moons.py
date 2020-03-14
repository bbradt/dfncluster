import numpy as np
from dfncluster.Dataset import SklearnDataset

KWARGS = dict(
    n_samples=10000,
    shuffle=False,
    noise=0.01,
    random_state=314159
)


class Moons:
    @staticmethod
    def make():
        dataset = SklearnDataset(dataset_name='make_moons', **KWARGS)
        dataset.save('data/SklearnDatasets/Moons/moons')
        return dataset


if __name__ == '__main__':
    dataset = Moons.make()
