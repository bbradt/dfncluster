import numpy as np
from dfncluster.Dataset import OpenNeuroDataset


class Ds000115:
    @staticmethod
    def make():
        dataset = OpenNeuroDataset("ds000115",
                                   directory='data/OpenNeuroDatasets/ds000115',
                                   modalities=['func'],
                                   series='*task-*.nii.gz',
                                   feature_columns=['task-letter0backtask_bold'],
                                   label_columns=['condit']
                                   )
        return dataset


if __name__ == '__main__':
    dataset = Ds000115.make()
