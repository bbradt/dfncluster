import copy
import numpy as np


def padded_stack(*args):
    '''Merge N-dimensional numpy arrays with different shapes

    '''
    ac = copy.deepcopy(list(args))
    max_dim = 0
    for x in args:
        if x.ndim > max_dim:
            max_dim = x.ndim
    shapes = []
    max_shape = []
    for i, x in enumerate(ac):
        while x.ndim < max_dim:
            x = x[np.newaxis, ...]
        for j, s in enumerate(x.shape):
            if j >= len(max_shape):
                max_shape.append(s)
            elif s > max_shape[j]:
                max_shape[j] = s
        ac[i] = x
        shapes.append(x.shape)
    print(max_shape)
    for i, (s, x) in enumerate(zip(shapes, ac)):
        diffs = []
        for s_b, s_a in zip(max_shape, s):
            diffs.append((0, max(s_b-s_a, 0)))
        ac[i] = np.pad(x, diffs)
    return np.squeeze(np.stack(ac, 0))
