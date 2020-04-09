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
    'distance_threshold'
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


class HierarchicalClusterer(Clusterer):
    @staticmethod
    def default_params():
        return dict(
            n_clusters = 5,
            affinity = 'euclidean',
            memory = None,
            #compute_full_tree,
            linkage = 'ward'
            #distance_threshold
        )

    def __init__(self, **kwargs):
        super(HierarchicalClusterer, self).__init__(**kwargs)
        self.model = skc.AgglomerativeClustering(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})
        

    def fit(self):
        self.labels_ = self.model.fit_predict(self.X) #fit
        self.assignments = self.labels_
        print((self.assignments).shape)
        print(np.unique(self.assignments))
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
            if metric in LABEL_METRICS.keys():
                results[metric] = LABEL_METRICS[metric](self.X, self.labels_)
        results['adjusted_rand_score'] = SCORE_METRICS['adjusted_rand_score'](self.Y[:,0], self.labels_)
        self.results = results
        return results

    def get_results_for_init(self):
        """
        Return own results in a dictionary, that maps to initialization for running
            a second time.

            This method also seems not necessary for hierarchical clustering, since it is a 
            hard clustering approach that doesn't require any initialization. Still
            implemented in order to smoothly run the dFNC pipeline.
        """
        
        return dict()
