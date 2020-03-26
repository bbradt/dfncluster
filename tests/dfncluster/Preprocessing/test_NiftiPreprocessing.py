import numpy as np
from dfncluster.Preprocessing import NiftiPreprocessing
from unittest.mock import MagicMock
DIM = 10
FAKE_ND = dict(
    d4=dict(
        input=np.ones((DIM, DIM, DIM, DIM)),
        output=np.ones((DIM*DIM*DIM, DIM))
    ),
    d3=dict(
        input=np.ones((DIM, DIM, DIM)),
        output=np.ones((DIM*DIM*DIM, 1))
    ),
    d2=dict(
        input=np.ones((DIM, DIM)),
        output=np.ones((DIM, DIM))
    )
)

REMOVE_MEAN = dict(
    input=np.ones((DIM, DIM)),
    output=np.zeros((DIM, DIM))
)

FAKE_SUBJECTS = [
    dict(
        data=np.array([
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0]
        ]).T,
        mask=np.array([
            [0, 0, 0, 1, 0, 0, 0]
        ]).T
    ),
    dict(
        data=np.array([
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0]
        ]).T,
        mask=np.array([
            [0, 0, 0, 1, 1, 0, 0]
        ]).T
    ),
    dict(
        data=np.array([
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0]
        ]).T,
        mask=np.array([
            [0, 0, 0, 0, 1, 0, 0]
        ]).T
    )
]

GROUP_MASK = dict(
    input=[s['mask'] for s in FAKE_SUBJECTS],
    hard=np.array([[0, 0, 0, 0, 0, 0, 0]]).T,
    soft=np.array([[0, 0, 0, 1, 1, 0, 0]]).T,
)


class TestNiftiPreprocessing:

    def test_flatten(self):
        for io in FAKE_ND.values():
            flattened = NiftiPreprocessing.flatten(io['input'])
            np.testing.assert_array_equal(flattened, io['output'])

    def test_remove_mean(self):
        demeaned = NiftiPreprocessing.remove_mean(REMOVE_MEAN['input'])
        np.testing.assert_array_equal(demeaned, REMOVE_MEAN['output'])

    def test_compute_subject_mask(self):
        for subject in FAKE_SUBJECTS:
            mask = NiftiPreprocessing.compute_subject_mask(subject['data'])
            np.testing.assert_array_equal(mask, subject['mask'])

    def test_compute_group_mask_hard(self):
        mask = NiftiPreprocessing.compute_group_mask(GROUP_MASK['input'], threshold=1)
        np.testing.assert_array_equal(mask, GROUP_MASK['hard'])

    def test_compute_group_mask_soft(self):
        mask = NiftiPreprocessing.compute_group_mask(GROUP_MASK['input'], threshold=0.33)
        np.testing.assert_array_equal(mask, GROUP_MASK['soft'])

    def test_apply_mask(self):
        for subject in FAKE_SUBJECTS:
            masked = NiftiPreprocessing.apply_mask(subject['data'], GROUP_MASK['hard'])
            assert(masked.shape == (np.sum(GROUP_MASK['hard']), subject['data'].shape[1]))
            masked = NiftiPreprocessing.apply_mask(subject['data'], GROUP_MASK['soft'])
            assert(masked.shape == (np.sum(GROUP_MASK['soft']), subject['data'].shape[1]))

    def test_apply(self):
        NiftiPreprocessing.flatten = MagicMock(return_value=FAKE_ND['d4']['output'])
        NiftiPreprocessing.compute_subject_mask = MagicMock(return_value=FAKE_ND['d4']['output'])
        NiftiPreprocessing.compute_group_mask = MagicMock(return_value=FAKE_ND['d4']['output'])
        NiftiPreprocessing.apply_mask = MagicMock(return_value=FAKE_ND['d4']['output'])
        NiftiPreprocessing.remove_mean = MagicMock(return_value=FAKE_ND['d4']['output'])
        obj = NiftiPreprocessing()
        result = obj.apply(np.array([FAKE_ND['d4']['input']]))
        NiftiPreprocessing.flatten.assert_called_once()
        NiftiPreprocessing.compute_subject_mask.assert_called_once()
        NiftiPreprocessing.compute_group_mask.assert_called_once()
        NiftiPreprocessing.apply_mask.assert_called_once()
        NiftiPreprocessing.remove_mean.assert_called_once()
        assert(hasattr(obj, 'global_mask'))
        assert(hasattr(obj, 'local_masks'))
        np.testing.assert_array_equal(result, np.array([FAKE_ND['d4']['output']]))
