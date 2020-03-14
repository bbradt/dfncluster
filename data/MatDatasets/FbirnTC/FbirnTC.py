import json
from dfncluster.Dataset import MatDataset


class FbirnTC:
    @staticmethod
    def make():
        dataset = MatDataset(filename="data/MatDatasets/FbirnTC/data.csv",
                             feature_columns=['ica_tc'],
                             label_columns=['diagnosis'])
        dataset.save('data/MatDatasets/FbirnTC/fbirn_tc')
        return dataset
