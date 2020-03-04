import json
from dfncluster.Dataset import MatDataset

if __name__ == '__main__':
    dataset = MatDataset(filename="data/MatDatasets/FbirnTC/data.csv",
                           feature_columns=['ica_tc'],
                           label_columns=['diagnosis'])
    dataset.save('data/MatDatasets/FbirnTC/fbirn_tc')
