import numpy as np
from dfncluster.Dataset import OpenNeuroDataset


class Ds000030:
    @staticmethod
    def make():
        dataset = OpenNeuroDataset("ds000030",
                                   directory='data/OpenNeuroDatasets/ds000030',
                                   feature_columns=['task-rest_bold'],
                                   label_columns=['diagnosis'])
        return dataset


if __name__ == '__main__':
    dataset = Ds000030.make()
