import os
import numpy as np
from copy import deepcopy
from unittest.mock import patch
from tempfile import TemporaryDirectory, TemporaryFile
from dfncluster.Dataset import FileFeatureDataset

PARAMETERS = dict(
    shuffle=True,
    filename='data.csv',
    feature_columns=['feature'],
    label_columns=['label'],
    loader=np.load
)
TEST_CLASS = FileFeatureDataset


class TestFileFeatureDataset:

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
        N_INSTANCES = 100
        test_features = np.random.randn(N_INSTANCES, 30)
        test_labels = np.random.randint(0, 2, size=(N_INSTANCES,))
        with TemporaryDirectory() as tempdir:
            full_fname = os.path.join(tempdir, PARAMETERS['filename'])
            with open(full_fname, 'w') as file:
                file.write('feature,label\n')
                for i in range(N_INSTANCES):
                    sub_features = test_features[i, ...]
                    fn = os.path.join(tempdir, 'sub%d.npy' % i)
                    np.save(fn, sub_features)
                    file.write('%s,%d\n' % (fn, test_labels[i]))
            params = deepcopy(PARAMETERS)
            params['filename'] = full_fname
            dataset = TEST_CLASS(**params)
            x, y = dataset.generate(**params)
            np.testing.assert_equal(x, test_features)
            np.testing.assert_equal(y, test_labels)
