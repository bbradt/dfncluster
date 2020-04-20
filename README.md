# dFNCluster

dFNCluster implements Dynamic Functional Network Connectivity (dFNC) with several clustering algorithms, and
actively compares the performance of classification under different clustering algorithms and hyper-parameters.

# Introduction
Schizophrenia is a chronic and serious mental disorder which affects how a person thinks, feels, and behaves. Although there have been many studies about psychological and behavioral manifestations of schizophrenia, neuroscientists have yet to determine a set of corresponding neurological biomarkers for this disorder. A functional magnetic resonance imaging (fMRI) can help determine non-invasive biomarkers for schizophrenia in brain function[[1]](#ref1)[[2]](#ref2) and one of the fMRI analysis technique called, dynamic functional network connectivity (dFNC)[[3]](#ref3)[[4]](#ref4), using K-Means clustering to characterize time-varying connectivity between functional networks. Researches have worked on finding correlation between schizophrenia and dFNC[[1]](#ref1)[[2]](#ref2)[[5]](#ref5)[[6]](#ref6), but little work has been done with the choice of clustering algorithm[[7]](#ref7). Therefore, in this project, we study how modifying the clustering technique in the dFNC pipeline can yield dynamic states from fMRI data that impact the accuracy of classifying schizophrenia[[8]](#ref8).

We experiment with DBSCAN, Hiearcharial Clustering, Gaussian Mixture Models, and Bayesian Gaussian Mixture Models clustering methods on subject connectivity matrices produced from fMRI data, and each algorithm's cluster assignments as features for SVMs, MLP, Nearest Neighbor, and other supervised classification algorithms to classify schizophrenia.

Section II describes the fMRI data used in our experimentation, while Section III summarizes the aforementioned clustering and classification algorithms used in the pipeline. Section IV compares the accuracy of these classifiers, along with presenting a series of charts that analyze the cluster assignments produced on the fMRI data.

# Section II: Data

# Section III: Methods

# Section IV: Results


### Gaussian Simulated Dataset 

![](images/sim_pre_clustering_AUC.png?raw=true)


### FBIRN Dataset 

![](images/fbirn_pre_clustering_AUC.png?raw=true)

### UCLA Dataset
![](images/ucla_pre_clustering_AUC.png?raw=true)

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


## References
<a name="ref1"></a> [1]. Eswar Damaraju et al. “Dynamic functional connectivity analysis reveals transient states of dyscon-nectivity in schizophrenia”. In:NeuroImage: Clinical5 (2014), pp. 298–308.<br>

<a name="ref2"></a> [2]. Mustafa S Salman et al. “Group ICA for identifying biomarkers in schizophrenia:‘Adaptive’networks viaspatially constrained ICA show more sensitivity to group differences than spatio-temporal regression”.In:NeuroImage: Clinical22 (2019), p. 101747.<br>

<a name="ref3"></a> [3]. Elena A Allen et al. “Tracking whole-brain connectivity dynamics in the resting state”. In:Cerebralcortex24.3 (2014), pp. 663–676.<br>

<a name="ref4"></a> [4]. D. Zhi et al., "Abnormal Dynamic Functional Network Connectivity and Graph Theoretical Analysis in Major Depressive Disorder," 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Honolulu, HI, 2018, pp. 558-561.<br>

<a name="ref5"></a> [5]. U Sakoglu, AM Michael, and VD Calhoun. “Classification of schizophrenia patients vs healthy controlswith dynamic functional network connectivity”. In:Neuroimage47.1 (2009), S39–41.<br>

<a name="ref6"></a> [6]. Unal  Sako ̆glu  et  al.  “A  method  for  evaluating  dynamic  functional  network  connectivity  and  task-modulation: application to schizophrenia”. In:Magnetic Resonance Materials in Physics, Biology andMedicine23.5-6 (2010), pp. 351–366.<br>

<a name="ref7"></a> [7]. V. M. Vergara, A. Abrol, F. A. Espinoza and V. D. Calhoun, "Selection of Efficient Clustering Index to Estimate the Number of Dynamic Brain States from Functional Network Connectivity*," 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Berlin, Germany, 2019, pp. 632-635.<br>

<a name="ref8"></a> [8]. D. K. Saha, A. Abrol, E. Damaraju, B. Rashid, S. M. Plis and V. D. Calhoun, “Classification As a Criterion to Select Model Order For Dynamic Functional Connectivity States in Rest-fMRI Data,” 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), Venice, Italy, 2019, pp. 1602-1605.<br>

<a name="ref9"></a> [9]. Pedregosa et al. “2.3. Clustering.” Scikit, scikit-learn.org/stable/modules/clustering.html.<br>

<a name="ref10"></a> [10]. Rashid, Barnaly, et al. “Classification of Schizophrenia and Bipolar Patients Using Static and Dynamic Resting-State FMRI Brain Connectivity.” NeuroImage, vol. 134, 2016, pp. 645–657., doi:10.1016/j.neuroimage.2016.04.051.



