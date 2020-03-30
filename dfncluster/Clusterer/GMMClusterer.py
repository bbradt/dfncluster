import sklearn.cluster as skc
from dfncluster.Clusterer import Clusterer
import sklearn.mixture
from sklearn.mixture import GaussianMixture
import numpy as np

ALLOWED_KWARGS = [
    'n_components',
    'covariance_type',
    'tol',
    'reg_covar',
    'max_iter',
    'n_init',
    'init_params',
    'weights_init',
    'means_init',
    'precisions_init',
    'random_state',
    'warm_start',
    'verbose',
    'verbose_interval'
]

import sklearn.metrics as skm
import numpy as np
import abc

def paired_wrapper(metric, agg=np.sum):
    """Wrap paired metrics with a numpy aggregation function
        Args:
            metric - function that returns a numpy ndarray
        Kwargs:
            agg - the aggregation function - returns a float
        Returns:
            function handle for the newly wrapped function
    """
    def wrapped(X,Y):
        return agg(metric(X,Y))
    return wrapped

SCORE_METRICS = dict(
        adjusted_rand_score=skm.cluster.adjusted_rand_score
)

LABEL_METRICS = dict(
    # calinksi_harabaz=skm.calinski_harabasz_score,
    davies_bouldin=skm.davies_bouldin_score,
    silhouette=skm.silhouette_score,
)

CENTROID_METRICS = dict(
    sum_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.sum),
    mean_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.mean),
    min_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.min),
    max_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.max),
    sum_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.sum),
    mean_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.mean),
    min_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.min),
    max_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.max),
)

class GMMClusterer(Clusterer):
    def __init__(self, **kwargs):
        super(GMMClusterer, self).__init__(**kwargs)
        if self.centroids is not None:
            kwargs['init'] = self.centroids
        self.model = sklearn.mixture.GaussianMixture(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        uniqueValues, occurCount = np.unique(self.Y, return_counts=True)
        self.model.fit(self.X, self.X)
        self.weights_ = self.model.weights_
        self.means_ = self.model.means_
        self.covariances_ = self.model.covariances_
        self.precisions_ = self.model.precisions_
        self.precisions_cholesky_ = self.model.precisions_cholesky_
        self.converged_ = self.model.converged_
        self.n_iter_ = self.model.n_iter_
        self.lower_bound_ = self.model.lower_bound_

    def predict(self):
        self.model.predict_ = self.model.predict(self.X)


    def evaluate(self):
        """
            Run evaluation metrics and save in self.results

            Return:
                results - dict<string,float> - dictionary of results for each metric
        """
        results = dict()
        for metric in self.metrics:
            print('Evaluating clustering with metric %s' % metric)
            if metric in CENTROID_METRICS.keys():
                results[metric] = CENTROID_METRICS[metric](self.X, self.centroids)
            elif metric in LABEL_METRICS.keys():
                results[metric] = LABEL_METRICS[metric](self.X, self.model.predict_)
        results['adjusted_rand_score'] = SCORE_METRICS['adjusted_rand_score'](self.Y[:,0], self.model.predict_)
        self.results = results
        return results

