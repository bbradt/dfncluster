import json
from dfncluster.Dataset import MatDataset


class OmegaSim:
    @staticmethod
    def make():
        dataset = MatDataset(filename="data/MatDatasets/OmegaSim/data.csv",
                             feature_columns=['ica_tc'],
                             label_columns=['diagnosis'])
        dataset.save('data/MatDatasets/OmegaSim/omega_sim')
        return dataset


if __name__ == '__main__':
    dataset = OmegaSim.make()
