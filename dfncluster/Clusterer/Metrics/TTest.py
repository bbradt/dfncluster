""""
    TTest
    Function evaluates the clustered dFNC assignment differences
    across all time windows between negative and and positive cases,
    (0, 1) (healthy control, schizophrenic).
"""

from scipy.stats import ttest_ind
import numpy as np


def t_test(features, labels, p_level=None):
    """
    t_test: generates p_values across all time domain values b/w
            classes
    Args:
        features: NxD matrix where N is number of subjects and D is
                  their corresponding cluster assignment over time
        labels: 1-D matrix of labels for healthy control and schizophrenia
        p_level: if set, return boolean array where true represent a significant
                 difference between classes i.e where p_value[i] < p_level
    Returns:
        pvalues: 1-D matrix of corresponding p-values between healthy control and
                  schizophrenia
    """
    pvalues = np.zeros(features.shape[1])
    for dim in range(features.shape[1]):
        pvalues[dim] = ttest_ind(
            features[labels == 0, dim], features[labels == 1, dim],
            equal_var=True)[1] # ignore t-statistic
    if p_level is None:
        return pvalues
    return abs(pvalues) < p_level

