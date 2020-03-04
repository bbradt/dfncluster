from dfncluster.Dataset import FNCDataset
from dfncluster.Clusterer import KMeansClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier

# Parameters for KMeans
kmeans_params = dict(
    init='k-means++',
    n_init=100,
    tol=1e-6,
    n_clusters=5,
    metrics=['silhouette'],
    verbose=0
)

# Load the Data Set
fbirn_data = FNCDataset.load('data/FNCDatasets/FbirnFNC/fbirn_fnc.npy')
subject_data, subject_labels = fbirn_data.get_subjects()

# Create the dFNC Runner
dfnc = dFNC(dataset=fbirn_data, clusterer=KMeansClusterer)

# Run it, passing KMeans Params
results, assignments = dfnc.run(**kmeans_params)

# Print results
print(results)
print(assignments, assignments.shape, fbirn_data.labels.shape)

poly = Polyssifier(assignments, subject_labels)
poly.build()
poly.run()