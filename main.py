from dfncluster.Dataset import MatDataset, SklearnDataset
from dfncluster.Clusterer import KMeansClusterer, BayesianGMMClusterer, DBSCANClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
from data.MatDatasets.FbirnTC.FbirnTC import FbirnTC
import os
import pdb

import matplotlib.pyplot as plt


if __name__=='__main__':

    # Parameters for KMeans
    # kmeans_params = dict(
    #     init='k-means++',
    #     n_init=100,
    #     tol=1e-6,
    #     n_clusters=5,
    #     metrics=['silhouette'],
    #     verbose=0
    # )

    
    params = DBSCANClusterer.default_params()
    filename = 'data/MatDatasets/FbirnTC/fbirn_tc.npy'

    # Load the Data Set
    if not os.path.exists(filename):
        print("Remaking data set on disk")
        FbirnTC.make()
    print("Reloading data set from disk")
    fbirn_data = MatDataset.load(filename)

    # Create the dFNC Runner
    dfnc = dFNC(
        dataset=fbirn_data,
        clusterer=DBSCANClusterer,
        window_size=22, time_index=1)

    # Run it, passing [KMeans, BayesGMM, DBSCAN] params
    print("Running dFNC with DBSCAN clustering")
    results, assignments = dfnc.run(**params)

    subject_data, subject_labels = dfnc.get_subjects()

    # Print results
    print(results)
    print(assignments, assignments.shape, fbirn_data.labels.shape)

    os.makedirs('results/polyssifier/FNCOnly', exist_ok=True)

    poly = Polyssifier(assignments,
                       subject_labels,
                       n_folds=10,
                       path='results/polyssifier',
                       project_name='FNCOnly',
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
