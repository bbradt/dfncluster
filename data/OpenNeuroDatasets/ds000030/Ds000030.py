import numpy as np
from dfncluster.Dataset import OpenNeuroDataset


class Ds000030:
    @staticmethod
    def make():
        dataset = OpenNeuroDataset("ds000030",
                                   directory='data/OpenNeuroDatasets/ds000030')
        return dataset


if __name__ == '__main__':
    dataset = Ds000030.make()
