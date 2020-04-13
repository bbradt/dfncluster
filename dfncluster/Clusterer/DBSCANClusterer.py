import sklearn.cluster as skc
from dfncluster.Clusterer import Clusterer
import sklearn.metrics as skm
import numpy as np
import scipy.cluster.hierarchy as sch
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
    'evaluate'
]

SCORE_METRICS = dict(
        adjusted_rand_score=skm.cluster.adjusted_rand_score
)

LABEL_METRICS = dict(
    # calinksi_harabaz=skm.calinski_harabasz_score,
    davies_bouldin=skm.davies_bouldin_score,
    silhouette=skm.silhouette_score,
)


class DBSCANClusterer(Clusterer):
    @staticmethod
    def default_params():
        return dict(
            eps=0.01,
            min_samples=3,
            metrics=[],
            evaluate=True,
            metric="correlation"
        )

    def __init__(self, **kwargs):
        super(DBSCANClusterer, self).__init__(**kwargs)
        self.model = skc.DBSCAN(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})

    def fit(self):
        self.model.fit(self.X, self.Y)
        self.assignments = self.model.labels_
        centroids = []
        for k in np.unique(self.assignments):
            samples = self.X[self.assignments == k, :]
            centroids.append(np.mean(samples,0))
        self.centroids = np.vstack(centroids)
        print(self.centroids.shape)

    def get_results_for_init(self):
        """Return own results in a dictionary, that maps to initialization for running
            a second time.

            NOTE: This method doesn't appear necessary for DBSCAN, since it is a 
            hard clustering approach that doesn't require any initialization. Still
            implemented in order to smoothly run the dFNC pipeline.
        """
        return dict()

    def evaluate(self):
        """
            Run evaluation metrics and save in self.results

            Return:
                results - dict<string,float> - dictionary of results for each metric
        """
        results = dict()
        for metric in self.metrics:
            print('Evaluating clustering with metric %s' % metric)
            if metric in LABEL_METRICS.keys():
                results[metric] = LABEL_METRICS[metric](self.X, self.model.labels_)
        results['adjusted_rand_score'] = SCORE_METRICS['adjusted_rand_score'](self.Y[:,0], self.model.labels_)
        self.results = results
        return results