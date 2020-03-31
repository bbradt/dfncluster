#  Internal Modules
from dfncluster.Dataset import MatDataset
from dfncluster.Clusterer import KMeansClusterer, BayesianGMMClusterer, GMMClusterer
from dfncluster.dFNC import dFNC
from dfncluster.Classifiers import Polyssifier
#  Internal Dataset Imports
from data.MatDatasets.FbirnTC.FbirnTC import FbirnTC
from data.MatDatasets.OmegaSim.OmegaSim import OmegaSim
from data.SklearnDatasets.Blobs import Blobs
from data.SklearnDatasets.Iris import Iris
from data.SklearnDatasets.Moons import Moons
from data.SklearnDatasets.Classification import Classification
# External Modules
import os
import argparse
import json

# Constants
DATA_ROOT = 'data'
DATASETS = dict(
    iris=Iris,
    moons=Moons,
    fbirn=FbirnTC,
    blobs=Blobs,
    classification=Classification,
    simtb=OmegaSim
)
CLUSTERERS = dict(
    kmeans=KMeansClusterer,
    bgmm=BayesianGMMClusterer,
    gmm=GMMClusterer,
    dbscan=None,
    hierarchical=None,
    vae=None
)


def parse_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fbirn", type=str, help="<str> the data set to use. Options are fbirn, simtb, iris, moons, blobs, classification")
    parser.add_argument("--dataset_file", default=None, type=str, help="<str> the .npy file used by a data set to load so that it is not remade each time")
    parser.add_argument("--clusterer", default="kmeans", type=str, help="<str> the clusterer to use. Options are kmeans, bgmm, gmm, dbscan")
    parser.add_argument("--window_size", default=22, type=int, help="<int> the size of the dFNC window")
    parser.add_argument("--time_index", default=1, type=int, help="<int> the dimension in which dFNC windows will be computed")
    parser.add_argument("--clusterer_params", default="{}", type=str, help="<str(dict)> dict to be loaded for classifier params(JSON)")
    parser.add_argument("--classifier_params", default="{}", type=str, help="<str(dict)> dict to be loaded for classifier params (JSON)")
    parser.add_argument("--outdir", default="FNCOnly", type=str, help="<str> Name of the results directory. Saving hierarchy is: results/<outdir>")
    parser.add_argument("--dfnc", default=True, type=bool, help="<bool> Do or do not run dFNC")
    parser.add_argument("--classify", default=True, type=bool, help="<bool> Do or do not do classification")
    parser.add_argument("-N", default=314, type=int, help="<int> Number of subjects (instances) to test")
    return parser.parse_args()


if __name__ == '__main__':

    """
    TODO: make param generation an iterable data structure
    to test mutiple clustering algorithms and corresponding
    hyperparamters, need to study grid search API.
    """

    args = parse_main_args()
    if args.clusterer not in CLUSTERERS.keys():
        raise(ValueError("The clusterer %s has not been added to main.py" % args.clusterer))
    result_dir = os.path.join('results', args.outdir)
    os.makedirs(result_dir, exist_ok=True)

    InputClusterer = CLUSTERERS[args.clusterer]

    # Add input params to params
    params = InputClusterer.default_params()
    input_params = json.loads(args.clusterer_params)
    for k, v in input_params:
        params[k] = v

    if args.dataset not in DATASETS.keys():
        raise(ValueError("The dataset %s is currently not supported in main.py" % args.dataset))
    InputDataset = DATASETS[args.dataset]

    print("Loading data set")
    if args.dataset_file is not None and os.path.exists(args.dataset_file):
        dataset = InputDataset.load(args.dataset_file)
    else:
        dataset = InputDataset.make()
    features = dataset.features
    labels = dataset.labels

    # Create the dFNC Runner
    if args.dfnc:
        dfnc = dFNC(
            dataset=dataset,
            clusterer=InputClusterer,
            window_size=args.window_size, time_index=args.time_index)

        # Run it, passing [KMeans, BayesGMM, GMM] params
        print("Running dFNC with KMeans clustering")
        results, assignments = dfnc.run(**params)

        subject_data, subject_labels = dfnc.get_subjects()
        # Print results

        print("dFNC Clustering Results")
        print(results)

    if args.classify:
        if args.dfnc:
            features = subject_data
            labels = subject_labels
        poly = Polyssifier(features,
                           labels,
                           n_folds=10,
                           path='results',
                           project_name=args.outdir,
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
