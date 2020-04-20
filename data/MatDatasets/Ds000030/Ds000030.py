import json
from dfncluster.Dataset import MatDataset
from dfncluster.dFNC import dFNC

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
    dfnc = dFNC(
        dataset=dataset,
        first_stage_algorithm=None,
        second_stage_algorithm=None,
        window_size=22,
        time_index=1)
    fnc_features, fnc_labels = dfnc.compute_windows()
    dfnc.visualize_clusters(fnc_features, fnc_labels, 'Features', 'data/examples/gauss_features.png',None)
