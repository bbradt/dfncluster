import os
import argparse
from data.SklearnDatasets import Iris, Classification, MNIST, Moons
from dfncluster.Classifiers import Polyssifier


DATASETS = dict(
    # iris=Iris.make,
    classification=Iris.make,
    # mnist=MNIST.make,
    moons=Moons.make
)


def main(dataset, n_folds):
    dataset = DATASETS[dataset.lower()]()
    os.makedirs('results/polyssifier/PolyssifierExample', exist_ok=True)
    poly = Polyssifier(dataset.features,
                       dataset.labels,
                       n_folds=n_folds,
                       path='results/polyssifier',
                       project_name='PolyssifierExample',
                       concurrency=1)
    poly.build()
    poly.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--kfolds", type=int, default=10)
    args = parser.parse_args()
    main(args.dataset, args.kfolds)
