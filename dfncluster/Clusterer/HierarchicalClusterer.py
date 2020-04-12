import sklearn.cluster as skc
import scipy.cluster.hierarchy as sch
from dfncluster.Clusterer import Clusterer
import sklearn.metrics as skm
import numpy as np

ALLOWED_KWARGS = [
    'n_clusters',
    'affinity',
    'memory',
    'connectivity',
    'compute_full_tree',
    'linkage',
    'distance_threshold',
    'random_state'
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
    # euclidean, manhattan, 
    sum_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.sum),
    mean_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.mean),
    min_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.min),
    max_euclid=paired_wrapper(skm.pairwise.paired_euclidean_distances, np.max),
    sum_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.sum),
    mean_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.mean),
    min_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.min),
    max_city=paired_wrapper(skm.pairwise.paired_manhattan_distances, np.max),
)



class HierarchicalClusterer(Clusterer):
    def __init__(self, **kwargs):
        super(HierarchicalClusterer, self).__init__(**kwargs)
        self.model = skc.AgglomerativeClustering(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})
        
    def fit(self):
        self.labels_ = self.model.fit_predict(self.X) #fit
        self.n_clusters_ = self.model.n_clusters_
        self.n_leaves_ = self.model.n_leaves_
        self.n_connected_components_ = self.model.n_connected_components_
        self.children_ = self.model.children_

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
                results[metric] = LABEL_METRICS[metric](self.X, self.labels_)
        results['adjusted_rand_score'] = SCORE_METRICS['adjusted_rand_score'](self.Y[:,0], self.labels_)
        self.results = results
        return results
