import os
import numpy as np
from copy import deepcopy
import scipy.io as sio
from unittest.mock import patch
from tempfile import TemporaryDirectory, TemporaryFile
import nibabel as nib
from nibabel.testing import data_path
from dfncluster.Dataset import NiftiDataset
from dfncluster.Dataset.NiftiDataset import load_wrapper

PARAMETERS = dict(
    shuffle=True,
    filename='data.csv',
    feature_columns=['feature'],
    label_columns=['label']
)
TEST_CLASS = NiftiDataset


class TestNiftiDataset:

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
        N_INSTANCES = 2
        example_file = os.path.join(data_path, 'example4d.nii.gz')
        test_features = np.stack([nib.load(example_file).dataobj for n in range(N_INSTANCES)])
        test_labels = np.random.randint(0, 2, size=(N_INSTANCES,))
        with TemporaryDirectory() as tempdir:
            full_fname = os.path.join(tempdir, PARAMETERS['filename'])
            with open(full_fname, 'w') as file:
                file.write('feature,label\n')
                for i in range(N_INSTANCES):
                    file.write('%s,%d\n' % (example_file, test_labels[i]))
            params = deepcopy(PARAMETERS)
            params['filename'] = full_fname
            dataset = TEST_CLASS(**params)
            x, y = dataset.generate(loader=load_wrapper, **params)
            np.testing.assert_equal(x, test_features)
            np.testing.assert_equal(y, test_labels)
