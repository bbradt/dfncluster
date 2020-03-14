import argparse
from data.SklearnDatasets import Iris, Classification, MNIST, Moons, Blobs
from dfncluster.Clusterer import KMeansClusterer, Clusterer


DATASETS = dict(
    iris=Iris.make,
    classification=Iris.make,
    mnist=MNIST.make,
    moons=Moons.make
)

METRICS = [
    'calinksi_harabaz',
    'davies_bouldin',
    'silhouette',
    # 'mean_euclid',
    # 'mean_city',
]


def main(dataset, metrics):
    dataset = DATASETS[dataset.lower()]()
    clusterer = KMeansClusterer(X=dataset.features, Y=dataset.labels, metrics=metrics, num_clusters=5)
    clusterer.fit()
    results = clusterer.evaluate()
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--metrics", type=str, default=",".join(METRICS))
    args = parser.parse_args()
    main(args.dataset, args.metrics.split(','))
