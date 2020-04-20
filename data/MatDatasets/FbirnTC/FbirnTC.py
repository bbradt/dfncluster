import json
from dfncluster.Dataset import MatDataset
from dfncluster.dFNC import dFNC

class FbirnTC:
    @staticmethod
    def make():
        dataset = MatDataset(filename="data/MatDatasets/FbirnTC/data.csv",
                             feature_columns=['ica_tc'],
                             label_columns=['diagnosis'])
        dataset.save('data/MatDatasets/FbirnTC/fbirn_tc')
        return dataset


if __name__ == '__main__':
    dataset = FbirnTC.make()
    dfnc = dFNC(
        dataset=dataset,
        first_stage_algorithm=None,
        second_stage_algorithm=None,
        window_size=22,
        time_index=0)
    fnc_features, fnc_labels = dfnc.compute_windows()
    dfnc.visualize_clusters(fnc_features, fnc_labels, 'Features', 'data/examples/fbirn_features.png',None)
