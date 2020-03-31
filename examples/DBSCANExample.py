import argparse
import sklearn.metrics as skm
import numpy as np
import scipy.cluster.hierarchy as sch
from data.SklearnDatasets import Iris, Classification, MNIST, Moons, Blobs
from dfncluster.Clusterer import DBSCANClusterer

DATASETS = dict(
    iris=Iris.make,
    classification=Classification.make,
    mnist=MNIST.make,
    moons=Moons.make,
    blobs=Blobs.make
)

METRICS = [
    'calinksi_harabaz',
    'davies_bouldin',
    'silhouette',
]

def main(dataset, metrics):
    dataset = DATASETS[dataset.lower()]()
    clusterer = DBSCANClusterer(X=dataset.features, Y=dataset.labels, metrics=metrics, eps=1, min_samples=1)
    clusterer.fit()
    results = clusterer.evaluate()
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--metrics", type=str, default=",".join(METRICS))
    args = parser.parse_args()
    main(args.dataset, args.metrics.split(','))