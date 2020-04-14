# dFNCluster

dFNCluster implements Dynamic Functional Network Connectivity (dFNC) with several clustering algorithms, and
actively compares the performance of classification under different clustering algorithms and hyper-parameters.

# Introduction
Schizophrenia is a chronic and serious mental disorder which affects how a person thinks, feels, and behaves. Although there have been many studies about psychological and behavioral manifestations of schizophrenia, neuroscientists have yet to determine a set of corresponding neurological biomarkers for this disorder. In the meanwhile, functional magnetic resonance imaging (fMRI) data can help determine non-invasive biomarkers for schizophrenia in brain function[[1]](#ref1)[[2]](#ref2) and one of the fMRI analysis is dynamic functional network connectivity (dFNC)[[3]](#ref3) using K-Means clustering to characterize time-varying connectivity between functional networks. Researches has worked on finding correlation between schizophrenia and dFNC[[2]][[4]][[5]][[6]],(#ref2)(#ref4)(#ref5)(#ref6) but little work has been done with the choice of clustering algorithm. Therefore, we propse a  novel  study  of  dFNC  applying  sophisticated clustering techniques to determine dynamic states, and also exploring classifiers with a goal of improving schizophrenia classification.

# Dataset
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

## UCLA



### Running on Subsets

To run on a subset of the simulated data set, you can either edit data.csv in the data directory, and rebuild,
or copy that directory under a new name, edit, rebuild and point main.py to the new data set.

## Prerequisites

First, install git submodules

```
git submodule update --init --recursive
```

This project has been tested in Python 3.6+

It is recommended you use a conda virtual environment.

```
conda env create -y --name dfncluster
```

and install requirements via pip

```
pip install -r requirements.txt
```

## Methods


## Results



## Other Examples

TODO (Brad): Make example run files, and writeups for other data sets, such as OpenNeuro

## Repostory Organization

TODO (Brad): Add a writeup explaining the repository organization
