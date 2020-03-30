from dfncluster.Dataset import FNCDataset
from dfncluster.Dataset import SklearnDataset
from dfncluster.Clusterer import KMeansClusterer
from dfncluster.Clusterer.DBSCANClusterer import DBSCANClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
from data.FNCDatasets.FbirnFNC import FbirnFNC
from data.SklearnDatasets.Moons import Moons
import os
import numpy as np
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

    
    filename = 'data/SklearnDatasets/Iris/Iris.npy'


    # Load the Data Set
#     # fbirn_data = FNCDataset.load('data/FNCDatasets/OmegaSim/omega_sim.npy')
#     pdb.set_trace()
    # fbirn_data = np.load('data/SklearnDatasets/Moons/Moons.npy')

    if not os.path.exists(filename):
        print("Remaking data set on disk")
        Moons.make()
    print("Reloading data set from disk")
    fbirn_data = SklearnDataset.load(filename)

    dbscan_params = dict(
        eps=1,
        min_samples=1,
        X=fbirn_data.features,
        Y=fbirn_data.labels
    )

    clusterer = DBSCANClusterer(**dbscan_params)

    clusterer.fit()

    X = clusterer.X
    labels_true = clusterer.Y
    labels = clusterer.model.labels_

    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clusterer.model.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # DBSCANClusterer(fbirn_data.)
    # subject_data, subject_labels = fbirn_data.get_subjects()

    # train_data, train_labels = fbirn_data.features, fbirn_data.labels 

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()



    # Create the dFNC Runner
    # dfnc = dFNC(dataset=fbirn_data, clusterer=DBSCANClusterer)

    # # Run it, passing KMeans Params
    # print("Running dFNC with KMeans clustering")
    # results, assignments = dfnc.run(**dbscan_params)

    # # Print results
    # print(results)
    # print(assignments, assignments.shape, fbirn_data.labels.shape)

    # os.makedirs('results/polyssifier/KMeans', exist_ok=True)

    # poly = Polyssifier(assignments,
    #                    subject_labels,
    #                    n_folds=10,
    #                    path='results/polyssifier',
    #                    project_name='KMeans',
    #                    concurrency=1)
    # poly.build()
    # poly.run()

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
