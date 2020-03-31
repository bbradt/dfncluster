from dfncluster.Dataset import MatDataset
from dfncluster.Clusterer import KMeansClusterer, BayesianGMMClusterer, GMMClusterer, DBSCANClusterer
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
import numpy as np
# Warning suppression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Constants
DATA_ROOT = 'data'
DATASETS = dict(
    fbirn=FbirnTC,
    simtb=OmegaSim,
)
DATASET_TYPES = dict(
    fbirn=MatDataset,
    simtb=MatDataset,
)
DATASET_FILE = dict(
    fbirn=os.path.join('data', 'MatDatasets', 'FbirnTC', 'fbirn_tc.npy'),
    simtb=os.path.join('data', 'MatDatasets', 'OmegaSim', 'omega_sim.npy'),
)
CLUSTERERS = dict(
    kmeans=KMeansClusterer,
    bgmm=BayesianGMMClusterer,
    gmm=GMMClusterer,
    dbscan=DBSCANClusterer,
    hierarchical=None,
    vae=None
)


def parse_main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fbirn", type=str,
                        help="<str> the data set to use. Options are fbirn, simtb, gaussian; DEFAULT=%s" % "fbirn")
    parser.add_argument("--remake_data", default=False, type=bool, help="<bool> whether or not to remake the data set; DEFAULT=%s" % False)
    parser.add_argument("--clusterer", default="kmeans", type=str,
                        help="<str> the clusterer to use. Options are kmeans, bgmm, gmm, dbscan; DEFAULT=%s" % "kmeans")
    parser.add_argument("--window_size", default=22, type=int, help="<int> the size of the dFNC window; DEFAULT=%s" % 22)
    parser.add_argument("--time_index", default=1, type=int, help="<int> the dimension in which dFNC windows will be computed; DEFAULT=%s" % 1)
    parser.add_argument("--clusterer_params", default="{}", type=str, help="<str(dict)> dict to be loaded for classifier params(JSON); DEFAULT=%s" % "\"{}\"")
    parser.add_argument("--classifier_params", default="{}", type=str, help="<str(dict)> dict to be loaded for classifier params (JSON); DEFAULT=%s" % "\"{}\"")
    parser.add_argument("--outdir", default="FNCOnly", type=str,
                        help="<str> Name of the results directory. Saving hierarchy is: results/<outdir>; DEFAULT=%s" % "FNCOnly")
    parser.add_argument("--dfnc", default=True, type=bool, help="<bool> Do or do not run dFNC; DEFAULT=%s" % True)
    parser.add_argument("--classify", default=True, type=bool, help="<bool> Do or do not do classification; DEFAULT=%s" % True)
    parser.add_argument("--subset_size", default=1.0, type=float, help="<float [0,1]> percentage of data to use; DEFAULT=1.0 (all data)")
    parser.add_argument("--dfnc_outfile", default="dfnc.npy", type=str, help="<str> The filename for saving dFNC results; DEFAULT=dfnc.npy")
    parser.add_argument("--seed", default=None,
                        help="<int> Seed for numpy RNG. Used for random generation of the data set, or for controlling randomness in Clusterings.; DEFAULT=None (do not use seed)",)
    parser.add_argument("--k", default=5, help="<int> number of folds for k-fold cross-validation")
    return parser.parse_args()


if __name__ == '__main__':

    """
    TODO: make param generation an iterable data structure
    to test mutiple clustering algorithms and corresponding
    hyperparamters, need to study grid search API.
    """

    args = parse_main_args()
    print("ARGS")
    print(args.__dict__)
    if args.clusterer not in CLUSTERERS.keys():
        raise(ValueError("The clusterer %s has not been added to main.py" % args.clusterer))
    result_dir = os.path.join('results', args.outdir)
    os.makedirs(result_dir, exist_ok=True)

    InputClusterer = CLUSTERERS[args.clusterer]

    # Add input params to params
    params = InputClusterer.default_params()
    input_params = json.loads(args.clusterer_params)
    for k, v in input_params.items():
        params[k] = v

    if args.dataset not in DATASETS.keys():
        raise(ValueError("The dataset %s is currently not supported in main.py" % args.dataset))
    InputDataset = DATASETS[args.dataset]

    print("Loading data set")
    if not os.path.exists(DATASET_FILE[args.dataset]) or args.remake_data:
        if args.seed is not None:
            np.random.seed(args.seed)
        dataset = InputDataset.make()
        dataset.save(DATASET_FILE[args.dataset])
    else:
        dataset = DATASET_TYPES[args.dataset].load(DATASET_FILE[args.dataset])
    sub_N = int(dataset.num_instances*args.subset_size)

    features = dataset.features[:sub_N, ...]
    labels = dataset.labels[:sub_N, ...]

    # Create the dFNC Runner
    if args.dfnc:
        if args.seed is not None:
            np.random.seed(args.seed)
        dfnc = dFNC(
            dataset=dataset,
            clusterer=InputClusterer,
            window_size=args.window_size, time_index=args.time_index)

        # Run it, passing [KMeans, BayesGMM, GMM] params
        print("Running dFNC with %s clustering" % args.clusterer)
        results, assignments = dfnc.run(**params)

        subject_data, subject_labels = dfnc.get_subjects()
        # Print results

        print("dFNC Clustering Results")
        print(results, assignments)
        print("Saving dFNC Results")
        dfnc.save(os.path.join('results', args.outdir, args.dfnc_outfile))

    if args.classify:
        if args.dfnc:
            features = assignments
            labels = subject_labels
        if args.seed is not None:
            np.random.seed(args.seed)
        poly = Polyssifier(features,
                           labels,
                           n_folds=5,
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
