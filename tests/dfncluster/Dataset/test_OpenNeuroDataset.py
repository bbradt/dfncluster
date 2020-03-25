import os
import numpy as np
from copy import deepcopy
import scipy.io as sio
from unittest.mock import patch
from tempfile import TemporaryDirectory, TemporaryFile
import nibabel as nib
from nibabel.testing import data_path
from dfncluster.Dataset import OpenNeuroDataset
from dfncluster.Dataset.NiftiDataset import load_wrapper

PARAMETERS = dict(
    shuffle_instances=True,
    filename='data.csv',
    feature_columns=['feature'],
    label_columns=['label'],
    pre_shuffle=False
)
TEST_CLASS = OpenNeuroDataset


class TestOpenNeuroDataset:

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
