import sklearn.cluster as skc
import scipy.cluster.hierarchy as sch
from dfncluster.Clusterer import Clusterer
import sklearn.metrics as skm
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sb

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


def plot_dendrogram(heir_instance, **kwargs):

    # create the counts of samples under each node to create linkage matrix
    counts = np.zeros(heir_instance.children_.shape[0])
    n_samples = len(heir_instance.labels_)
    for i, merge in enumerate(heir_instance.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    distances = sch.linkage(heir_instance.X, method=heir_instance.model.linkage)

    linkage_matrix = np.column_stack([heir_instance.children_, distances[:,2], counts[:, np.newaxis]]).astype(float)
    sch.dendrogram(linkage_matrix, **kwargs)
    plt.savefig(heir_instance.dendrogram_name)


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
            compute_full_tree = 'auto',
            linkage = 'ward',
            distance_threshold = None
        )

    def __init__(self, dendogram_filename="results/dendogram.png", **kwargs):
        super(HierarchicalClusterer, self).__init__(**kwargs)
        self.model = skc.AgglomerativeClustering(**{k:v for k,v in kwargs.items() if k in ALLOWED_KWARGS})
        
        self.dendrogram_name = dendogram_filename
        

    def fit(self):
        self.labels_ = self.model.fit_predict(self.X) #fit
        self.assignments = self.labels_
        self.n_clusters_ = self.model.n_clusters_
        self.n_leaves_ = self.model.n_leaves_
        self.n_connected_components_ = self.model.n_connected_components_
        self.children_ = self.model.children_
        centroids = []
        for k in np.unique(self.assignments):
            samples = self.X[self.assignments == k, :]
            centroids.append(np.mean(samples, 0))
        self.centroids = np.vstack(centroids)
        try:
            plot_dendrogram(self, truncate_mode='level', p=self.n_clusters_)
        except AttributeError:
            pass
        

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
    

    def evaluate_k(self, ks, filename):
        distortions = []
        silhouettes = []
        for k in ks:
            print("Evaluating KMeans with %d clusters" % k)
            clf = skc.AgglomerativeClustering(n_clusters=k)
            clf.fit(self.X, self.Y)
            centroids = []
            for k in np.unique(clf.labels_):
                samples = self.X[clf.labels_ == k, :]
                centroids.append(np.mean(samples, 0))
            centroids = np.vstack(centroids)
            distortions.append(sum(np.min(cdist(self.X, centroids, 
                                'correlation'),axis=1)) / self.X.shape[0]) 
            silhouettes.append(skm.silhouette_score(self.X, clf.fit_predict(self.X)))
        sb.set()
        fig, ax = plt.subplots()
        ax.plot(ks, distortions)
        ax.set_title('Elbow Criterion - Agglomerative')
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