# dFNCluster

dFNCluster implements Dynamic Functional Network Connectivity (dFNC) with several clustering algorithms, and
actively compares the performance of classification under different clustering algorithms and hyper-parameters.

# Prerequisites

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

# Examples

To run with the pre-computed ICA Timecourses, run the following

first, untar the simulated data
```
cd data/FNCDatasets/OmegaSim
tar -xzf subjects.tar.gz
```

back in the dfncluster directory, build the simulations data set, which serializes
the data set object as a pickled npy file

```
PYTHONPATH=. python data/FNCDatasets/OmegaSim/make.py
```

And run the main function, which performs dFNC analysis, and classification
using cluster assignments, and raw states for comparison
```
python main.py
```

## Running on Subsets

To run on a subset of the simulated data set, you can either edit data.csv in the OmegaSim directory, and rebuild,
or copy that directory under a new name, edit, rebuild and point main.py to the new data set.

## Other Examples

TODO (Brad): Make example run files, and writeups for other data sets, such as OpenNeuro

# Repostory Organization

TODO (Brad): Add a writeup explaining the repository organization
