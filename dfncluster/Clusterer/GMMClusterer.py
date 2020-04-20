import abc
import sklearn.metrics as skm
import sklearn.cluster as skc
from dfncluster.Clusterer import Clusterer
import sklearn.mixture
from sklearn.mixture import GaussianMixture
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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
    @staticmethod
    def default_params():
        return dict(
            n_components=5,
            covariance_type='full',
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
            n_init=1,
            init_params='kmeans',
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=None,
            warm_start=False,
            verbose_interval=10,
            metrics=['silhouette'],
            verbose=0
        )

    def __init__(self, **kwargs):
        super(GMMClusterer, self).__init__(**kwargs)
        self.model = sklearn.mixture.GaussianMixture(**{k: v for k, v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        self.model.fit(self.X, self.Y)
        self.assignments = self.model.predict(self.X)
        self.model.predict_ = self.assignments
        self.pred_proba = self.model.predict_proba(self.X)
        self.centroids = self.model.means_
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
        results['adjusted_rand_score'] = SCORE_METRICS['adjusted_rand_score'](self.Y[:, 0], self.model.predict_)
        self.results = results
        return results

    def evaluate_k(self, ks, filename):
        distortions = []
        silhouettes = []
        for k in ks:
            print("Evaluating KMeans with %d clusters" % k)
            clf = sklearn.mixture.GaussianMixture(n_components=k, random_state=0)
            clf.fit(self.X, self.Y)
            distortions.append(sum(np.min(cdist(self.X, clf.means_, 
                                'correlation'),axis=1)) / self.X.shape[0]) 
            silhouettes.append(skm.silhouette_score(self.X, clf.predict(self.X)))
        sb.set()
        fig, ax = plt.subplots()
        ax.plot(ks, distortions)
        ax.set_title('Elbow Criterion - GMM')
        ax.set_ylabel('Correlation Distortion')
        ax.set_xlabel('Number of Components')

        ax2 = ax.twinx()
        ax2.plot(ks, silhouettes, color='r')
        ax2.set_ylabel('Silhouette Score')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.savefig(filename, bbox_inches="tight")
        print('saving in %s' % filename)
        plt.close()


    def get_results_for_init(self):
        """Return own results in a dictionary, that maps to initialization for running
            a second time.
        """
        n_components = self.centroids.shape[0]
        return dict(means_init=self.centroids)
