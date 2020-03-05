import json
from dfncluster.Dataset import FNCDataset

if __name__ == '__main__':
    dataset = FNCDataset(filename="data/FNCDatasets/OmegaSim/data.csv",
                         time_index=1,
                           feature_columns=['ica_tc'],
                           label_columns=['diagnosis'],
                           shuffle=False)
    dataset.save('data/FNCDatasets/OmegaSim/omega_sim')
