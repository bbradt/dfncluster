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
    def __init__(self, **kwargs):
        super(KMeansClusterer, self).__init__(**kwargs)
        if self.centroids is not None:
            kwargs['init'] = self.centroids
        self.model = skc.KMeans(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        self.model.fit(self.X, self.Y)
        self.centroids = self.model.cluster_centers_
        self.assignments = self.model.labels_
