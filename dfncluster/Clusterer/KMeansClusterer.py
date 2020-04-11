import sklearn.cluster as skc
from dfncluster.Clusterer import Clusterer

ALLOWED_KWARGS = [
    'n_clusters',
    'init',
    'n_init',
    'max_iter',
    'tol',
    'precompute_distances',
    'verbose',
    'random_state',
    'copy_x',
    'n_jobs',
    'algorithm'
]


class KMeansClusterer(Clusterer):
    @staticmethod
    def default_params():
        return dict(
            init='k-means++',
            n_init=100,
            tol=1e-6,
            n_clusters=5,
            metrics=['silhouette'],
            verbose=0,
            n_jobs=16
        )

    def __init__(self, initialization={}, **kwargs):
        super(KMeansClusterer, self).__init__(**kwargs)
        for k, v in initialization.items():
            kwargs[k] = v
        self.model = skc.KMeans(**{k: v for k, v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        self.model.fit(self.X, self.Y)
        self.centroids = self.model.cluster_centers_
        self.assignments = self.model.labels_

    def get_results_for_init(self):
        """Return own results in a dictionary, that maps to initialization for running
            a second time.
        """
        return dict(init=self.centroids)
