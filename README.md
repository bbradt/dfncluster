# Introduction

Although psychological and behavioral manifestations of schizophrenia are well-studied, neuroscientists have
yet to determine a set of corresponding neurological biomarkers. One
fMRI analysis technique, called dynamic functional network connectivity  (dFNC) [1], uses K-Means clustering
to characterize time-varying connectivity between functional networks to identify schizophrenia biomarkers. Although dFNC has been used to determine biomarkers in the past [14, 12, 13, 5] little attention has been given to choice of
clustering technique. In this project, we study how modifying the clustering technique in the dFNC pipeline
can yield dynamic states from fMRI data impact the accuracy of identifying schizophrenia with varying classifiers.

We specifically apply DBSCAN, Hiearcharial Clustering, Gaussian Mixture Models, and Bayesian Gaussian Mixture Models clustering methods on time-varying connectivity fMRI data and use the output cluster classifications as features for SVMs, MLP, Nearest Neighbor, and other supervised classification algorithms to classify schizophrenia. 

Section II delves into the data that is used in this project's experiments, while Section III summarizes the different algorithms used on both in the clustering and classification end of our experiments. Section IV compares the accuracy from using clustering and classification algorithms for schizophrenia diagnosis, along with presenting a series of charts that analyze the clustering states produced from the selected fMRI data.


# Section II: Data

# Section III: Methods

# Section IV: Results

# dFNCluster

dFNCluster implements Dynamic Functional Network Connectivity (dFNC) with several clustering algorithms, and
actively compares the performance of classification under different clustering algorithms and hyper-parameters.

## Prerequisites

First, install git submodules

```
git submodule update --init --recursive
```

This project has been tested in Python 3.6+

It is recommended you use a conda virtual environment.

```
conda create -y --name dfncluster
```

and install requirements via pip

```
pip install -r requirements.txt
```

## Running the Code

You can run `main.py` with the arguments given below, or look at them by running `python main.py --help`

```
usage: main.py [-h] [--dataset DATASET] [--remake_data REMAKE_DATA]
               [--clusterer CLUSTERER] [--window_size WINDOW_SIZE]
               [--time_index TIME_INDEX] [--clusterer_params CLUSTERER_PARAMS]
               [--classifier_params CLASSIFIER_PARAMS] [--outdir OUTDIR]
               [--dfnc DFNC] [--classify CLASSIFY] [--subset_size SUBSET_SIZE]
               [--dfnc_outfile DFNC_OUTFILE] [--seed SEED] [--k K]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     <str> the data set to use. Options are fbirn, simtb,
                        gaussian; DEFAULT=fbirn
  --remake_data REMAKE_DATA
                        <bool> whether or not to remake the data set;
                        DEFAULT=False
  --clusterer CLUSTERER
                        <str> the clusterer to use. Options are kmeans, bgmm,
                        gmm, dbscan; DEFAULT=kmeans
  --window_size WINDOW_SIZE
                        <int> the size of the dFNC window; DEFAULT=22
  --time_index TIME_INDEX
                        <int> the dimension in which dFNC windows will be
                        computed; DEFAULT=1
  --clusterer_params CLUSTERER_PARAMS
                        <str(dict)> dict to be loaded for classifier
                        params(JSON); DEFAULT="{}"
  --classifier_params CLASSIFIER_PARAMS
                        <str(dict)> dict to be loaded for classifier params
                        (JSON); DEFAULT="{}"
  --outdir OUTDIR       <str> Name of the results directory. Saving hierarchy
                        is: results/<outdir>; DEFAULT=FNCOnly
  --dfnc DFNC           <bool> Do or do not run dFNC; DEFAULT=True
  --classify CLASSIFY   <bool> Do or do not do classification; DEFAULT=True
  --subset_size SUBSET_SIZE
                        <float [0,1]> percentage of data to use; DEFAULT=1.0
                        (all data)
  --dfnc_outfile DFNC_OUTFILE
                        <str> The filename for saving dFNC results;
                        DEFAULT=dfnc.npy
  --seed SEED           <int> Seed for numpy RNG. Used for random generation
                        of the data set, or for controlling randomness in
                        Clusterings.; DEFAULT=None (do not use seed)
  --k K                 <int> number of folds for k-fold cross-validation
```

## Examples

## Sklearn Datasets

To generate a data set from SKlearn for testing purposes, you can generate one of the datasets in `data/SklearnDatasets`.

For example, the moons data set can be generated as follows:

```
PYTHONPATH=. python data/SklearnDatasets/Moons/Moons.py
```

which will save `moons.npy` in the `data/SklearnDatasets/Moons` directory.

The following datasets have been included as examples:

* Moons
* Classification (sklearn.datasets.make_classification)
* Blobs (sklearn.datasets.make_blobs)
* MNIST (sklearn.datasets.fetch_openml(name='mnist_...)
* Iris (sklearn.datasets.load_iris)

## Fbirn Data

To run with the pre-computed ICA Timecourses from real data, run the following

first, untar the data
```
cd data/MatDatasets/FbirnTC
tar -xzf subjects.tar.gz
```

back in the dfncluster directory, build the data set, which serializes
the data set object as a pickled npy file

```
PYTHONPATH=. python data/MatDatasets/FbirnTC/FbirnTC.py
```

And run the main function, which performs dFNC analysis, and classification
using cluster assignments, and raw states for comparison
```
PYTHONPATH=. python main.py
```


## SIMTB Data

To run with the pre-computed ICA Timecourses, run the following

first, untar the simulated data
```
cd data/MatDatasets/OmegaSim
tar -xzf subjects.tar.gz
```

back in the dfncluster directory, build the simulations data set, which serializes
the data set object as a pickled npy file

```
PYTHONPATH=. python data/MatDatasets/OmegaSim/OmegaSim.py
```

### Running on Subsets

To run on a subset of the simulated data set, you can either edit data.csv in the data directory, and rebuild,
or copy that directory under a new name, edit, rebuild and point main.py to the new data set.

## Other Examples

TODO (Brad): Make example run files, and writeups for other data sets, such as OpenNeuro

## Repostory Organization

TODO (Brad): Add a writeup explaining the repository organization
