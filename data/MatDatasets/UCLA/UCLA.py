import json
from dfncluster.Dataset import MatDataset
from dfncluster.dFNC import dFNC

class UCLA:
    @staticmethod
    def make():
        dataset = MatDataset(filename="data/MatDatasets/UCLA/data.csv",
                             feature_columns=['task-rest_bold'],
                             label_columns=['diagnosis'],
                             shuffle_instances=False)
        dataset.save('data/MatDatasets/UCLA/ucla')
        return dataset


if __name__ == '__main__':
    dataset = UCLA.make()
    dfnc = dFNC(
        dataset=dataset,
        first_stage_algorithm=None,
        second_stage_algorithm=None,
        window_size=22,
        time_index=0)
    fnc_features, fnc_labels = dfnc.compute_windows()
    dfnc.visualize_clusters(fnc_features, fnc_labels, 'Features', 'data/examples/ucla_features.png',None)
