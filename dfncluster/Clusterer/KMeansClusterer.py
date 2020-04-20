import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.cluster as skc
from sklearn.metrics import silhouette_score
from dfncluster.Clusterer import Clusterer
from scipy.spatial.distance import cdist

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

    def evaluate_k(self, ks, filename):
        distortions = []
        silhouettes = []
        for k in ks:
            print("Evaluating KMeans with %d clusters" % k)
            clf = skc.KMeans(n_clusters=k, random_state=0, n_jobs=32)
            clf.fit(self.X, self.Y)
            distortions.append(sum(np.min(cdist(self.X, clf.cluster_centers_, 
                                'correlation'),axis=1)) / self.X.shape[0]) 
            silhouettes.append(silhouette_score(self.X, clf.predict(self.X)))
        sb.set()
        fig, ax = plt.subplots()
        ax.plot(ks, distortions)
        ax.set_title('Elbow Criterion - KMeans')
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
        n_clusters = self.centroids.shape[0]
        return dict(init=self.centroids, n_clusters=n_clusters)
