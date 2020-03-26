import numpy as np
from dfncluster.utils import padded_stack


N = 10
DIM = 10
MAX = 20
TEST = dict(
    input=[np.zeros((DIM, np.random.randint(1, MAX))) for n in range(N-1)] + [np.zeros((DIM, MAX))],
    output=np.zeros((N, DIM, MAX))
)


def test_padded_stack():
    result = padded_stack(*TEST['input'])
    np.testing.assert_array_equal(TEST['output'], result)
