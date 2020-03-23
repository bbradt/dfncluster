from dfncluster.Dataset.MatDataset import load_wrapper
from dfncluster.Dataset import SklearnDataset
from tempfile import TemporaryDirectory, TemporaryFile
from unittest.mock import patch
import scipy.io as sio
from copy import deepcopy
import numpy as np
import os
import sklearn.datasets as skd
DISALLOWED_FUNCTIONS = [
    'clear_data_home',
    'dump_svmlight_file',
    'load_svmlight_file',
    'load_svmlight_files',
    'get_data_home',
    'load_files',
    'load_sample_image',
    'load_sample_images',
    'make_biclusters',
    'make_checkerboard',
    'make_spd_matrix',
    'make_low_rank_matrix',
    'make_sparse_coded_signal',
    'make_sparse_spd_matrix',
    'make_spd_matrix'
]
SKLEARN_DATASETS = {n: getattr(skd, n) for n in dir(skd) if (n[0] != '_' and 'fetch' not in n and n not in DISALLOWED_FUNCTIONS)}


PARAMETERS = dict(
)
TEST_CLASS = SklearnDataset


class TestSklearnDataset:

    def test_init(self):
        with patch.object(TEST_CLASS, 'generate') as mock_generate:
            mock_generate.return_value = (np.array([[], []]), np.array([[], []]))
            dataset = TEST_CLASS(**PARAMETERS)

            assert(type(dataset) is TEST_CLASS)
            assert(hasattr(dataset, 'features'))
            assert(hasattr(dataset, 'labels'))
            assert(hasattr(dataset, 'num_instances'))
            assert(hasattr(dataset, 'num_features'))
            assert(hasattr(dataset, 'num_labels'))
            assert(hasattr(dataset, 'idx'))
            assert(hasattr(dataset, 'unique_labels'))
            assert(hasattr(dataset, 'num_unique_labels'))
            assert(hasattr(dataset, 'label_indices'))
            mock_generate.assert_called_once()

    def test_generate(self):
        for skd_name, skd_meth in SKLEARN_DATASETS.items():
            np.random.seed(123)
            dataset = TEST_CLASS(dataset_name=skd_name, shuffle_instances=False, **PARAMETERS)
            try:
                test_features, test_labels = skd_meth(return_X_y=True)
                x, y = dataset.generate(return_X_y=True, dataset_method=skd_meth)
            except Exception:
                np.random.seed(123)
                test_features, test_labels = skd_meth()
                np.random.seed(123)
                x, y = dataset.generate(dataset_method=skd_meth)
            np.testing.assert_equal(x, test_features)
            np.testing.assert_equal(y, test_labels)
            if test_labels.ndim == 1:
                test_labels = test_labels.reshape(np.shape(test_labels)[0], 1)
            np.testing.assert_equal(dataset.features, test_features)
            np.testing.assert_equal(dataset.labels, test_labels)
