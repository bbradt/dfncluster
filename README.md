# Introduction

Although psychological and behavioral manifestations of schizophrenia are well-studied, neuroscientists have
yet to determine a set of corresponding neurological biomarkers. One
fMRI analysis technique, called dynamic functional network connectivity  (dFNC) [1], uses K-Means clustering
to characterize time-varying connectivity between functional networks to identify schizophrenia biomarkers. Although dFNC has been used to determine biomarkers in the past [14, 12, 13, 5] little attention has been given to choice of
clustering technique. In this project, we study how modifying the clustering technique in the dFNC pipeline
can yield dynamic states from fMRI data that impact the accuracy of classifying schizophrenia.

We experiment with DBSCAN, Hiearcharial Clustering, Gaussian Mixture Models, and Bayesian Gaussian Mixture Models clustering methods on subject connectivity matrices produced from fMRI data, and each algorithm's cluster assignments as features for SVMs, MLP, Nearest Neighbor, and other supervised classification algorithms to classify schizophrenia. 

Section II describes the fMRI data used in our experimentation, while Section III summarizes the aforementioned clustering and classification algorithms used in the pipeline. Section IV compares the accuracy of these classifiers, along with presenting a series of charts that analyze the cluster assignments produced on the fMRI data.

# Section II: Data

# Section III: Methods

# Section IV: Results


### Gaussian Simulated Dataset 

![](images/sim_pre_clustering_AUC.png?raw=true)

Above plot shows "Area Under Curve" of various classifiers on simulated Gaussian Data before clustering where SVM seems to be performing the best followed by "Ada Boost" and "Gradient Boost" while "Bernoulli Naive Bayes" seems to be performing the worst.

![](results/gauss_betas_accuracy.png?raw=true)

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data using beta features. Accuracy has improved a lot in cmoparison to the previous case above without clusterer and without using beta feature say for example from 0.4 to 0.6 for KMeans and likewise for other clusterer. Multilayer percepteon classifier shows some improvement in GMM over KMeans and Passive Aggressive Classifier in BGMM shows some improvement over KMeans clustering.


### FBIRN Dataset 

![](images/fbirn_pre_clustering_AUC.png?raw=true)

Above plot shows "Area Under Curve" of various classifiers on FBirn Data before clustering where "Random Forest" seems to be performing the best followed by "Gradient Boost" and "Multi Layer Perceptron" while "Decision Tree" and "Bernoulli Naive Bayes" seems to be performing the worst.


![](results/fbirn_assignments_accuracy.png?raw=true)

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data without using beta features. Accuracy has improved a lot in cmoparison to the previous case above without clusterer and without using beta feature say for example from 0.5 to 0.7 for KMeans with "Multi Layer Perceptron" and likewise for other clusterer. Multilayer percepteon classifier shows some improvement in GMM over KMeans and "Logistic Regression" and "Passive Aggressive Classifier" in BGMM shows some improvement over KMeans clustering.

![](results/fbirn_betas_accuracy.png?raw=true)

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data using beta features. Accuracy has improved a lot in cmoparison to the previous case above with clusterer but without using beta feature say for example from 0.7 to 0.9 for KMeans with "Multi Layer Perceptron" and likewise for other clusterer. "Random Forest" and "Extra Trees" classifier shows some improvement in GMM over KMeans and "Random Forest", "Extra Trees"  and "Perceptron" Classifier in BGMM shows some improvement over KMeans clustering.

### UCLA Dataset

![](images/ucla_pre_clustering_AUC.png?raw=true)

Above plot shows "Area Under Curve" of various classifiers on UCLA Data before clustering where SVM seems to be performing the best followed by "Multilayer Perceptron" and "Gradient Boost" while "Gaussian Process" and "Decision Tree" seems to be performing the worst.

![](results/ucla_betas_accuracy.png?raw=true)

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data using beta features. Accuracy has improved a lot in cmoparison to the previous case above without clusterer and without using beta feature say for example from 0.7 to 0.9 for KMeans with "Multi Layer Perceptron" and likewise for other clusterer. Almost all the classifiers in all the clusterer shows improvement over KMeans clustering.

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

## References:

[1]. Rashid, Barnaly, et al. “Classification of Schizophrenia and Bipolar Patients Using Static and Dynamic Resting-State FMRI Brain Connectivity.” NeuroImage, vol. 134, 2016, pp. 645–657., doi:10.1016/j.neuroimage.2016.04.051.

[2]. Damaraju, E., et al. “Dynamic Functional Connectivity Analysis Reveals Transient States of Dysconnectivity in Schizophrenia.” NeuroImage: Clinical, vol. 5, 2014, pp. 298–308., doi:10.1016/j.nicl.2014.07.003.

[3]. V. M. Vergara, A. Abrol, F. A. Espinoza and V. D. Calhoun, "Selection of Efficient Clustering Index to Estimate the Number of Dynamic Brain States from Functional Network Connectivity*," 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Berlin, Germany, 2019, pp. 632-635.

[4]. D. K. Saha, A. Abrol, E. Damaraju, B. Rashid, S. M. Plis and V. D. Calhoun, "Classification As a Criterion to Select Model Order For Dynamic Functional Connectivity States in Rest-fMRI Data," 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), Venice, Italy, 2019, pp. 1602-1605.

[5]. Elena A. Allen, Eswar Damaraju, Sergey M. Plis, Erik B. Erhardt, Tom Eichele, Vince D. Calhoun, Tracking Whole-Brain Connectivity Dynamics in the Resting State, Cerebral Cortex, Volume 24, Issue 3, March 2014, Pages 663–676, https://doi.org/10.1093/cercor/bhs352

[6]. D. Zhi et al., "Abnormal Dynamic Functional Network Connectivity and Graph Theoretical Analysis in Major Depressive Disorder," 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Honolulu, HI, 2018, pp. 558-561.

[7]. Pedregosa et al. “2.3. Clustering.” Scikit, scikit-learn.org/stable/modules/clustering.html.
