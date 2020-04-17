import argparse
from data.SklearnDatasets import Iris, Classification, MNIST, Moons, Blobs
from dfncluster.Clusterer import HierarchicalClusterer

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

LINKAGE = ['single', 'complete', 'average', 'ward']

def main(dataset, metrics):
    dataset = DATASETS[dataset.lower()]()
    for link in LINKAGE:
        clusterer = HierarchicalClusterer(X=dataset.features, Y=dataset.labels, metrics=metrics, linkage=link)
        clusterer.fit()
        results = clusterer.evaluate()
        print("Linkage:", link, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--metrics", type=str, default=",".join(METRICS))
    args = parser.parse_args()
    main(args.dataset, args.metrics.split(','))