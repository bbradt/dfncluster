import argparse
from data.SklearnDatasets import Iris, Classification, MNIST, Moons, Blobs
from dfncluster.Clusterer import GMMClusterer, Clusterer
import numpy as np

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
    unique_val, occur_cnt = np.unique(dataset.labels, return_counts=True)
    clusterer = GMMClusterer(X=dataset.features, Y=dataset.labels, metrics=metrics, n_components=len(unique_val))
    clusterer.fit()
    results = clusterer.evaluate()
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--metrics", type=str, default=",".join(METRICS))
    args = parser.parse_args()
    main(args.dataset, args.metrics.split(','))
