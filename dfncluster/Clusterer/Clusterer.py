"""
    Clusterer
    Abstract class for clustering wrappers.
    Every clusterer is assumed to have the following fields:
        centroids
        assignments
        model
        metrics
        results
        X
        Y
        params
    Sub-Classes of Clusterer should override the fit() method,
        and define any additional fields in their own __init__
        functions.

    This class also defines the different metrics for cluster evaluation
"""

import sklearn.metrics as skm
import sklearn.model_selection as skms
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
    def wrapped(X, Y):
        return agg(metric(X, Y))
    return wrapped


LABEL_METRICS = dict(
    calinksi_harabaz=skm.calinski_harabaz_score,
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


class Clusterer:
    def __init__(self, metrics=[], X=[], Y=[], centroids=None, initialization={}, param_grid=None, **kwargs):
        """
            metrics - list<str> - list of metrics to use for evaluation
            X - ndarray<float> - NxD features array
            Y - ndarray<float> - Nx1 labels array
            centroids - ndarray<float> - KxD array
            **kwargs -- additional kwargs
        """
        self.centroids = centroids
        self.assignments = np.ones((X.shape[0], ))
        self.model = None
        self.metrics = metrics
        self.results = dict()
        self.X = X
        self.Y = Y
        self.params = kwargs
        self.params['metrics'] = metrics
        self.param_grid = param_grid

    @staticmethod
    @abc.abstractmethod
    def default_params():
        """
            Abstact run time parameter generation method. Helps
            encapsulates clustering algorithm withing each clustering
            class to ease integration testing.
        """
        pass

    @abc.abstractmethod
    def fit(self):
        """
            Abstract fit function which runs the clustering algorithm.
            Should always assign self.centroids and self.assignments.
            Can assign other fields in subclasses.
        """
        self.centroids = []
        self.assignments = []

    def fit_grid(self):
        if self.param_grid is not None:
            clf = skms.GridSearchCV(self.model, self.param_grid)
            clf.fit(self.X, self.Y)
            return clf
        else:
            return self.model

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
                results[metric] = LABEL_METRICS[metric](self.X, self.assignments)
        self.results = results
        return results

    def save(self, filename):
        """
            Save the clusterer, serializing centroids and assignments.

            Returns:
                None
            Saves:
                filename.npy of dict with params, centroids, assignment, results, model name
        """
        package = self.params
        package['centroids'] = self.centroids
        package['assignments'] = self.assignments
        package['results'] = self.results
        package['model'] = self.model.__name__
        np.save(filename, package, allow_pickle=True)
