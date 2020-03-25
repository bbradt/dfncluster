from dfncluster.Dataset import FNCDataset
from dfncluster.Clusterer import KMeansClusterer
from dfncluster.Clusterer import DBSCANClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
from data.FNCDatasets.FbirnFNC import FbirnFNC
import os
import numpy as np
import pdb

if __name__=='__main__':
    pdb.set_trace()

    # Parameters for KMeans

    pdb.set_trace() 
    # kmeans_params = dict(
    #     init='k-means++',
    #     n_init=100,
    #     tol=1e-6,
    #     n_clusters=5,
    #     metrics=['silhouette'],
    #     verbose=0
    # )

    dbscan_params = dict(
        eps=0.5,
        min_samples=5,
    )
    filename = 'data/FNCDatasets/FbirnFNC/fbirn_fnc.npy'


    # Load the Data Set
#     # fbirn_data = FNCDataset.load('data/FNCDatasets/OmegaSim/omega_sim.npy')
#     pdb.set_trace()
#     fbirn_data = np.load('data/SklearnDatasets/Moons/Moons.npy')
    if not os.path.exists(filename):
        print("Remaking data set on disk")
        FbirnFNC.make()
    print("Reloading data set from disk")
    fbirn_data = FNCDataset.load('data/FNCDatasets/FbirnFNC/fbirn_fnc.npy')
    subject_data, subject_labels = fbirn_data.get_subjects()

    pdb.set_trace()

    # Create the dFNC Runner
    dfnc = dFNC(dataset=fbirn_data, clusterer=DBSCANClusterer)

    # Run it, passing KMeans Params
    print("Running dFNC with KMeans clustering")
    results, assignments = dfnc.run(**kmeans_params)

    # Print results
    print(results)
    print(assignments, assignments.shape, fbirn_data.labels.shape)

    os.makedirs('results/polyssifier/KMeans', exist_ok=True)

    poly = Polyssifier(assignments,
                       subject_labels,
                       n_folds=10,
                       path='results/polyssifier',
                       project_name='KMeans',
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
