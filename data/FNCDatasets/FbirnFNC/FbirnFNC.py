import json
from dfncluster.Dataset import FNCDataset


class FbirnFNC:
    @staticmethod
    def make():
        dataset = FNCDataset(filename="data/FNCDatasets/FbirnFNC/data.csv",
                             feature_columns=['ica_tc'],
                             label_columns=['diagnosis'],
                             shuffle=False)
        dataset.save('data/FNCDatasets/FbirnFNC/fbirn_fnc')
        return dataset
