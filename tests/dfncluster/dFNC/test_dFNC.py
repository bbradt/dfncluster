# TODO: These tests are incomplete
import os
import numpy as np
from copy import deepcopy
import scipy.io as sio
from unittest.mock import patch
from tempfile import TemporaryDirectory, TemporaryFile
from dfncluster.Dataset import FNCDataset
from dfncluster.Dataset.MatDataset import load_wrapper

PARAMETERS = dict(
    shuffle_instances=True,
    filename='data.csv',
    feature_columns=['feature'],
    label_columns=['label'],
    pre_shuffle=False
)
TEST_CLASS = FNCDataset

