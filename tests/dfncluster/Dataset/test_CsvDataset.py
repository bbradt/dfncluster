import os
import numpy as np
from copy import deepcopy
from unittest.mock import patch
from tempfile import TemporaryDirectory, TemporaryFile
from tests.dfncluster.Dataset.test_Dataset import TestDataset
from dfncluster.Dataset import CsvDataset

PARAMETERS = dict(
    shuffle=True,
    filename='data.csv',
    feature_columns=['feature'],
    label_columns=['label']
)
TEST_CLASS = CsvDataset


class CsvTestDataset:

    def test_init(self):
        with patch.object(TEST_CLASS, 'generate') as mock_generate:
            with patch.object(TEST_CLASS, 'shuffle') as mock_shuffle:
                mock_generate.return_value = (np.array([[], []]), np.array([[], []]))
                mock_shuffle.return_value = None
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
                mock_shuffle.assert_called_once()

    def test_generate(self):
        feature_1 = [0 for f in PARAMETERS['feature_columns']]
        label_1 = [0 for f in PARAMETERS['label_columns']]
        feature_2 = [1 for f in PARAMETERS['feature_columns']]
        label_2 = [1 for f in PARAMETERS['label_columns']]
        test_features = np.vstack((np.array(feature_1), np.array(feature_2)))
        test_labels = np.vstack((np.array(label_1), np.array(label_2)))
        with TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, PARAMETERS['filename']), 'w') as file:
                file.write(','.join(PARAMETERS['feature_columns']) + ',' + ','.join(PARAMETERS['label_columns']) + '\n')
                file.write(','.join(feature_1) + ',' + ','.join(label_1) + '\n')
                file.write(','.join(feature_2) + ',' + ','.join(label_2) + '\n')
            dataset = TEST_CLASS(**PARAMETERS)
            x, y = dataset.generate(**PARAMETERS)
            np.testing.assert_equal(x, test_features)
            np.testing.assert_equal(y, test_labels)
