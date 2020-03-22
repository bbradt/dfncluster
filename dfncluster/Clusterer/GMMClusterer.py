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
    calinksi_harabaz=skm.calinski_harabasz_score,
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
#         print("GMM:init:kwargs:",kwargs)
        if self.centroids is not None:
            kwargs['init'] = self.centroids
#         print("GMM:init:kwargs:",kwargs)
        self.model = sklearn.mixture.GaussianMixture(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
#         print("GMM:fit:self",self)
#         print("GMM:fit:self.X.shape:",self.X.shape)
#         print("GMM:fit:self.X:",self.X)
        uniqueValues, occurCount = np.unique(self.Y, return_counts=True)
#         print("Unique Values : " , uniqueValues)
#         print("Occurrence Count : ", occurCount)
#         print("GMM:fit:self.Y.shape:",self.Y.shape)
#         print("GMM:fit:self.Y:",self.Y)
#         print("GMM:fit:self.model:",self.model)
        self.model.fit(self.X, self.X)
#         print("self.model.weights_.shape:",self.model.weights_.shape)
#         print("self.model.weights_:",self.model.weights_)
        self.weights_ = self.model.weights_
#         print("self.model.means_.shape:",self.model.means_.shape)
#         print("self.model.means_:",self.model.means_)
        self.means_ = self.model.means_
#         print("self.model.covariances_.shape:",self.model.covariances_.shape)
#         print("self.model.covariances_:",self.model.covariances_)
        self.covariances_ = self.model.covariances_
#         print("self.model.precisions_.shape:",self.model.precisions_.shape)
#         print("self.model.precisions_:",self.model.precisions_)
        self.precisions_ = self.model.precisions_
#         print("self.model.precisions_cholesky_.shape:",self.model.precisions_cholesky_.shape)
#         print("self.model.precisions_cholesky_:",self.model.precisions_cholesky_)
        self.precisions_cholesky_ = self.model.precisions_cholesky_
#         print("self.model.converged_:",self.model.converged_)
        self.converged_ = self.model.converged_
#         print("self.model.n_iter_:",self.model.n_iter_)
        self.n_iter_ = self.model.n_iter_
#         print("self.model.lower_bound_:",self.model.lower_bound_)
        self.lower_bound_ = self.model.lower_bound_

    def predict(self):
#         print("GMM:predict:self",self)
#         print("GMM:predict:self.X.shape:",self.X.shape)
        self.model.predict_ = self.model.predict(self.X)
#         print("type(self.model.predict_):",type(self.model.predict_.shape))
#         print("self.model.predict_.shape:",self.model.predict_.shape)
#         print("self.model.predict_:",self.model.predict_)


    def evaluate(self):
        """
            Run evaluation metrics and save in self.results

            Return:
                results - dict<string,float> - dictionary of results for each metric
        """
#         print("GMM-evaluate-1:")
        results = dict()
        for metric in self.metrics:
            print('Evaluating clustering with metric %s' % metric)
            if metric in CENTROID_METRICS.keys():
                results[metric] = CENTROID_METRICS[metric](self.X, self.centroids)
            elif metric in LABEL_METRICS.keys():
#                 print("metric:",metric)
#                 print("LABEL_METRICS[metric]:",LABEL_METRICS[metric])
#                 print("self.assignments.shape:",self.assignments.shape)
#                 print("self.assignments:",self.assignments)
                results[metric] = LABEL_METRICS[metric](self.X, self.model.predict_)
        results['adjusted_rand_score'] = SCORE_METRICS['adjusted_rand_score'](self.Y[:,0], self.model.predict_)
        self.results = results
        return results