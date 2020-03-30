from dfncluster.Dataset import MatDataset
from dfncluster.Clusterer import KMeansClusterer, BayesianGMMClusterer, GMMClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
from data.MatDatasets.FbirnTC.FbirnTC import FbirnTC
import os


if __name__ == '__main__':

    """
    TODO: make param generation an iterable data structure
    to test mutiple clustering algorithms and corresponding
    hyperparamters, need to study grid search API.
    """
    params = KMeansClusterer.default_params()
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
        clusterer=KMeansClusterer,
        window_size=22, time_index=1)

    # Run it, passing [KMeans, BayesGMM] params
    print("Running dFNC with KMeans clustering")
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
