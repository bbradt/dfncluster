import os
import numpy as np
from copy import deepcopy
from unittest.mock import patch
from tempfile import TemporaryDirectory

from dfncluster.Dataset import Dataset

PARAMETERS = dict(
    shuffle_instances=True
)
TEST_CLASS = Dataset


class TestDataset:

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

    def test_split(self):
        dataset = TEST_CLASS(**PARAMETERS)
        x_train, y_train, x_test, y_test = dataset.split(0.5)
        assert(x_train.shape[0] == int(dataset.num_instances/2))
        assert(y_train.shape[0] == int(dataset.num_instances/2))
        assert(x_test.shape[0] == int(dataset.num_instances/2))
        assert(y_test.shape[0] == int(dataset.num_instances/2))

    def test_generate(self):
        dataset = TEST_CLASS(**PARAMETERS)
        print("f", dataset.features, "l", dataset.labels)
        print(dataset.generate)
        x, y = dataset.generate(**PARAMETERS)
        test = np.array([[0], [1]])
        print("x", x, "y", y, "test", test)
        np.testing.assert_equal(x, test)
        np.testing.assert_equal(y, test)

    def test_shuffle(self):
        params = PARAMETERS
        params['shuffle'] = False
        dataset = TEST_CLASS(**params)
        initial_idx = deepcopy(dataset.idx)
        initial_features = dataset.features
        initial_labels = dataset.labels
        np.random.seed(123)
        dataset.shuffle()
        np.random.seed(123)
        np.random.shuffle(initial_idx)
        np.testing.assert_equal(dataset.idx, initial_idx)
        np.testing.assert_equal(dataset.features, initial_features[initial_idx, ...])
        np.testing.assert_equal(dataset.labels, initial_labels[initial_idx, ...])

    def test_save(self):
        dataset = TEST_CLASS(**PARAMETERS)
        with TemporaryDirectory() as tempdir:
            prefix = os.path.join(tempdir, 'test')
            dataset.save(prefix=prefix)
            filename = prefix+'.npy'
            assert(os.path.exists(filename))

    def test_load(self):
        dataset = TEST_CLASS(**PARAMETERS)
        with TemporaryDirectory() as tempdir:
            prefix = os.path.join(tempdir, 'test')
            dataset.save(prefix=prefix)
            filename = prefix+'.npy'
            test = TEST_CLASS.load(filename)
            assert(type(test) == TEST_CLASS)
