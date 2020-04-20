import json
from dfncluster.Dataset import GaussianConnectivityDataset
from dfncluster.dFNC import dFNC

class TestGCDataset:
    @staticmethod
    def make():
        dataset = GaussianConnectivityDataset()
        dataset.save('data/GaussianConnectivityDatasets/TestGCDataset/test_gc')
        return dataset


if __name__ == '__main__':
    dataset = TestGCDataset.make()
    dfnc = dFNC(
        dataset=dataset,
        first_stage_algorithm=None,
        second_stage_algorithm=None,
        window_size=22,
        time_index=1)
    fnc_features, fnc_labels = dfnc.compute_windows()
    dfnc.visualize_clusters(fnc_features, fnc_labels, 'Features', 'data/examples/gauss_features.png',None)
