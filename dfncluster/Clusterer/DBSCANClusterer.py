import sklearn.cluster as skc
from dfncluster.Clusterer import Clusterer
import pdb 

ALLOWED_KWARGS = [
    'eps',
    'min_samples',
    'metric',
    'metric_params',
    'algorithm',
    'leaf_size',
    'p',
    'n_jobs',
]

class DBSCANClusterer(Clusterer):
    def __init__(self, **kwargs):
        super(DBSCANClusterer, self).__init__(**kwargs)
        self.model = skc.DBSCAN(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        self.model.fit(self.X, self.Y)
        self.assignments = self.model.labels_