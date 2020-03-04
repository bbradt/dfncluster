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

```
python main.py
```

which will run dFNC with KMeans clustering and classification with Polyssifier