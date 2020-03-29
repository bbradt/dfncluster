import sklearn.mixture as skm
from dfncluster.Clusterer import Clusterer

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
    def __init__(self, **kwargs):
        super(BayesianGMMClusterer, self).__init__(**kwargs)
        if self.centroids is not None:
            kwargs['init_params'] = self.centroids
        self.model = skm.BayesianGaussianMixture(
            **{k: v for k, v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        self.assignments = self.model.fit_predict(self.X, self.Y)
        self.centroids = self.model.means_
