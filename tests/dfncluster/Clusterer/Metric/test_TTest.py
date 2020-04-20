from dfncluster.Clusterer.Metrics.TTest import t_test
import numpy as np


def test_main():

    standard = np.random.rand(100, 100)

    # all features across labels are different
    features = np.concatenate((standard, 10 * standard), axis=0)
    labels = np.concatenate((np.zeros(100), np.ones(100)))
    p_values = t_test(features, labels, p_level=5e-2)
    assert(np.count_nonzero(p_values) == features.shape[1])

    # all features across labels are identical
    features = np.concatenate((standard, standard), axis=0)
    p_values = t_test(features, labels, p_level=5e-2)

    # can detect isolated differences
    features[0:100, 10] = -23
    p_values = t_test(features, labels, p_level=5e-2)
    assert (np.count_nonzero(p_values) == 1)


def test_multiple_label_inputs():
    standard = np.random.rand(100, 100)

    # all features across labels are different
    features = np.concatenate((standard, 10 * standard), axis=0)
    labels = np.concatenate((np.zeros(100), np.ones(100)))

    labels_temp = labels.reshape((labels.shape[0], 1))
    p_values = t_test(features, labels_temp, p_level=5e-2)
    assert (np.count_nonzero(p_values) == features.shape[1])

    labels_temp = labels.reshape((1, labels.shape[0]))
    p_values = t_test(features, labels_temp, p_level=5e-2)
    assert (np.count_nonzero(p_values) == features.shape[1])