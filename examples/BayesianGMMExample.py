import argparse
from data.SklearnDatasets import Iris, Classification, MNIST, Moons, Blobs
from dfncluster.Clusterer import BayesianGMMClusterer, Clusterer

DATASETS = dict(
    iris=Iris.make,
    mnist=MNIST.make,
    moons=Moons.make
)

METRICS = [
    'calinksi_harabaz',
    'davies_bouldin',
    'silhouette',
]


def main(dataset, metrics):
    dataset = DATASETS[dataset.lower()]()
    clusterer = BayesianGMMClusterer(
        X=dataset.features, Y=dataset.labels,
        metrics=metrics,
        n_components=5,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=1e0)
    clusterer.fit()
    results = clusterer.evaluate()
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--metrics", type=str, default=",".join(METRICS))
    args = parser.parse_args()
    main(args.dataset, args.metrics.split(","))
