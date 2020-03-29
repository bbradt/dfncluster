from dfncluster.Dataset import MatDataset
from dfncluster.Clusterer import KMeansClusterer, GMMClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
from data.MatDatasets.FbirnTC.FbirnTC import FbirnTC
import os
import numpy as np

if __name__ == '__main__':

    # Parameters for GMM
    gmm_params = dict(
        n_components=2,
        covariance_type='diag',
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params='kmeans',
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose_interval=10,
        metrics=['silhouette'],
        verbose=0
    )
    filename = 'data/MatDatasets/FbirnTC/fbirn_tc.npy'

    # Load the Data Set
    if not os.path.exists(filename):
        print("Remaking data set on disk")
        FbirnTC.make()
    print("Reloading data set from disk")
    fbirn_data = MatDataset.load(filename)

    # Create the dFNC Runner
    dfnc = dFNC(dataset=fbirn_data, clusterer=GMMClusterer, window_size=22, time_index=1)

    # Run it, passing KMeans Params
    print("Running dFNC with GMM clustering")
    results, assignments = dfnc.run(**gmm_params)

    subject_data, subject_labels = dfnc.get_subjects()

    # Print results
    print(results)
    #print(assignments, assignments.shape, fbirn_data.labels.shape)

    os.makedirs('results/polyssifier/GMM', exist_ok=True)

    poly = Polyssifier(assignments,
                       subject_labels,
                       n_folds=10,
                       path='results/polyssifier',
                       project_name='GMM',
                       concurrency=1)
    poly.build()
    poly.run()

    """
    os.makedirs('results/polyssifier/FNCOnly', exist_ok=True)
  
    poly = Polyssifier(subject_data.reshape(subject_data.shape[0],np.prod(subject_data.shape[1:])),
                       subject_labels,
                       n_folds=10,
                       path='results/polyssifier',
                       project_name='FNCOnly',
                       concurrency=1)
    poly.build()
    poly.run()
    """
