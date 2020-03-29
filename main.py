from dfncluster.Dataset import MatDataset
from dfncluster.Clusterer import KMeansClusterer, BayesianGMMClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
from data.MatDatasets.FbirnTC.FbirnTC import FbirnTC
import os
import numpy as np

if __name__ == '__main__':

    # Parameters for KMeans
    params = dict(
        n_components=5,         # give a high number and allow alpha to reduce
        init_params='kmeans',   # use kmeans to set initial centers
        covariance_type='diag', # assume features are not independent
        n_init=1,               # number of initializations to perform
        weight_concentration_prior_type='dirichlet_process', # stick breaking cluster generation
        weight_concentration_prior=1. / 5, # default alpha weight
        metrics=['silhouette']
    )
    filename = 'data/MatDatasets/FbirnTC/fbirn_tc.npy'

    # Load the Data Set
    if not os.path.exists(filename):
        print("Remaking data set on disk")
        FbirnTC.make()
    print("Reloading data set from disk")
    fbirn_data = MatDataset.load(filename)

    # Create the dFNC Runner
    dfnc = dFNC(dataset=fbirn_data, clusterer=BayesianGMMClusterer, window_size=22, time_index=1)

    # Run it, passing KMeans Params
    print("Running dFNC with KMeans clustering")
    results, assignments = dfnc.run(**params)

    subject_data, subject_labels = dfnc.get_subjects()

    # Print results
    print(results)
    #print(assignments, assignments.shape, fbirn_data.labels.shape)

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
