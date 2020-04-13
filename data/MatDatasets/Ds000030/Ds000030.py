import json
from dfncluster.Dataset import MatDataset


class Ds000030:
    @staticmethod
    def make():
        dataset = MatDataset(filename="results/ica/ds000030/derivatives/data.csv",
                             feature_columns=['filename'],
                             label_columns=['diagnosis'])
        dataset.save('data/MatDatasets/Ds000030/ds000030')
        return dataset


if __name__ == '__main__':
    dataset = Ds000030.make()
