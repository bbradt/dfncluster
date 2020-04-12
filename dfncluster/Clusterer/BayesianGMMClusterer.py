import numpy as np
import sklearn.mixture as skm
from dfncluster.Clusterer import Clusterer
import seaborn as sb
import matplotlib.pyplot as plt

ALLOWED_KWARGS = [
    'n_components',
    'covariance_type',
    'tol',
    'reg_covar',
    'max_iter',
    'n_init',
    'init_params',
    'weight_concentration_prior_type',
    'weight_concentration_prior',
    'mean_precision_prior',
    'mean_prior',
    'degrees_of_freedom_prior',
    'covariance_prior',
    'random_state',
    'warm_start'
]


class BayesianGMMClusterer(Clusterer):
    @staticmethod
    def default_params():
        return dict(
            n_components=4,         # give a high number and allow alpha to reduce
            init_params='kmeans',   # use kmeans to set initial centers
            covariance_type='full',  # assume features are not independent, makes this a memeory hog :(
            n_init=1,               # number of initializations to perform
            weight_concentration_prior_type='dirichlet_process',  # stick breaking cluster generation
            weight_concentration_prior=1. / 5,  # default alpha weight
            metrics=['silhouette']
        )

    def __init__(self, **kwargs):
        super(BayesianGMMClusterer, self).__init__(**kwargs)
        self.model = skm.BayesianGaussianMixture(
            **{k: v for k, v in kwargs.items() if k in ALLOWED_KWARGS})

    def evaluate_k(self, ks, filename):
        distortions = []
        for k in ks:
            print("Evaluating KMeans with %d clusters" % k)
            clf = skm.BayesianGaussianMixture(n_components=k, random_state=0)
            clf.fit(self.X, self.Y)
            distortions.append(clf.score(self.X))
        sb.set()
        fig, ax = plt.subplots()
        ax.plot(ks, distortions)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


    def fit(self):
        self.assignments = self.model.fit_predict(self.X, self.Y)
        self.centroids = self.model.means_

    def get_results_for_init(self):
        """Return own results in a dictionary, that maps to initialization for running
            a second time.
        """
        n_components = self.centroids.shape[0]
        return dict(means_init=self.centroids, n_components=n_components)
