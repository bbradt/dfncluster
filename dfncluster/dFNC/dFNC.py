import numpy as np

class dFNC:
    def __init__(self, dataset=None, clusterer=None, **kwargs):
        """kwargs:
            dataset     FNCDataset  Some FNC Dataset
            clusterer   Clusterer   KMeansClusterer
        """
        self.dataset = dataset
        self.clusterer = clusterer
        self.results = []
    
    def run(self, **kwargs):
        exemplar_clusterer = self.clusterer(X=self.dataset.exemplars['x'], Y=self.dataset.exemplars['y'], **kwargs)
        exemplar_clusterer.fit()        
        cluster_instance = self.clusterer(X=self.dataset.features, Y=self.dataset.labels, centroids=exemplar_clusterer.centroids, **kwargs)
        cluster_instance.fit()
        cluster_instance.evaluate()
        assignments = self.reassign_to_subjects(cluster_instance.assignments, 
                                                self.dataset.subjects)
        return cluster_instance.results, assignments

    def reassign_to_subjects(self, cluster_assigments, subjects):
        reassigned = []
        for i in range(max(subjects)+1):
            reassigned[i] = cluster_assigments[subjects==i]
        return np.array(reassigned)

    def line_search(self, line_params=dict(param1=[]), **kwargs):
        """ Vary a particular parameter, and get clustering results when checking that parameter.
        """
        results = dict()
        assignments = dict()
        for param_name in line_params.keys():
            results[param_name] = dict()
            assignments[param_name] = dict()
            for param_val in line_params[param_name]:
                kwargs[param_name] = param_val
                results[param_name][param_val], assignments[param_name][param_val] = self.run(**kwargs)
        return results, assignments
  