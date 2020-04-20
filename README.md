# Introduction

Although psychological and behavioral manifestations of schizophrenia are well-studied, neuroscientists have
yet to determine a set of corresponding neurological biomarkers. One
fMRI analysis technique, called dynamic functional network connectivity  (dFNC) [1], uses K-Means clustering
to characterize time-varying connectivity between functional networks to identify schizophrenia biomarkers. Although dFNC has been used to determine biomarkers in the past [14, 12, 13, 5] little attention has been given to choice of
clustering technique. In this project, we study how modifying the clustering technique in the dFNC pipeline
can yield dynamic states from fMRI data that impact the accuracy of classifying schizophrenia.

We experimented with DBSCAN, Hiearcharial Clustering, Gaussian Mixture Models, and Bayesian Gaussian Mixture Models clustering methods on subject connectivity matrices produced from fMRI data, and each algorithm's cluster assignments as features for SVMs, MLP, Nearest Neighbor, and other supervised classification algorithms to classify schizophrenia.

Section II describes the fMRI data used in our experimentation, while Section III summarizes the aforementioned clustering and classification algorithms used in the pipeline. Section IV compares the accuracy of these classifiers, along with presenting a series of charts that analyze the cluster assignments produced on the fMRI data.

# Section II: Data

All datasets used in this project are derivatives of fMRI data (functional magnetic resonance imaging), which is data measuring brain activity to track a patient's thought processs over a predefined time period.

### Gaussian Simulated Dataset 

We derive our own synthetic dataset in order to test out our clusterers and classifiers on a simulated dataset for sanity checking their implementation. This Gaussian Simulated dataset follows .... (need more info here -> how should we write about this one?)

### FBIRN Dataset 

We use derivatives from the Phase 3 Dataset of FBIRN (Functional Biomedical Infromatics Research Network Data Repository), which specifically focuses on brain activity maps from patients with schizophrenia. This dataset includes 186 healthy controls and 176 indivduals from schizophrenia from around the United States. Subject participants in this dataset are between the ages of 18-62. 

### UCLA Dataset

We also use derivatives from the UCLA Consortium for Neuropsychiatric Phenomics archives, which includes neuroimages for roughly 272 participants. The subject population consists of roughly 272 healthy controls, as well as participants with a diagnosis of schizophrenia (50 subjects). Subject participants in this dataset range from 21 to 50 years 


### Pre-processing Pipeline

<Briefly describe here how used functional MRI processing, PCA, and ICA to reduce dataset to subject connectivity matrices?>

# Section III: Methods

<img width="50%" src="results/dfnc_pipeline(1).png?raw=True" />


# Section IV: Results & Discussion


## Gaussian Simulated Dataset 

| Clustering Algorithm | Multilayer Perceptron | Nearest Neighbors | SVM             | Random Forest | Extra Trees   | Gradient Boost  | Logistic Regression | Passive Aggressive Classifier | Perceptron      | Gaussian Process | Ada Boost         | Voting            | Bernoulli Naive Bayes | Bagging           | Decision Tree     |
| -------------------- | --------------------- | ----------------- | --------------- | ------------- | ------------- | --------------- | ------------------- | ----------------------------- | --------------- | ---------------- | ----------------- | ----------------- | --------------------- | ----------------- | ----------------- |
| kmeans               | 0.972 ± 0.026         | 0.96 ± 0.021      | 0.972 ± 0.023   | 0.962 ± 0.027 | 0.966 ± 0.024 | 0.954 ± 0.031   | 0.971 ± 0.022       | 0.974 ± 0.02                  | *0.947 ± 0.062* | 0.955 ± 0.027    | 0.957 ± 0.032     | 0.92 ± 0.037      | 0.948 ± 0.022         | 0.941 ± 0.035     | *0.938 ± 0.036*   |
| gmm                  | 0.963 ± 0.028         | 0.954 ± 0.03      | 0.972 ± 0.025   | 0.967 ± 0.022 | 0.976 ± 0.017 | 0.928 ± 0.046   | 0.962 ± 0.028       | 0.97 ± 0.024                  | 0.962 ± 0.028   | 0.952 ± 0.029    | 0.955 ± 0.031     | 0.92 ± 0.045      | 0.938 ± 0.028         | 0.917 ± 0.036     | 0.923 ± 0.033     |
| bgmm                 | 0.966 ± 0.023         | 0.949 ± 0.028     | *0.974 ± 0.027* | 0.966 ± 0.026 | 0.963 ± 0.028 | 0.962 ± 0.03    | 0.974 ± 0.02        | 0.972 ± 0.024                 | 0.969 ± 0.029   | 0.955 ± 0.029    | 0.958 ± 0.029     | 0.923 ± 0.045     | 0.949 ± 0.028         | 0.931 ± 0.043     | 0.933 ± 0.026     |
| dbscan               | 0.971 ± 0.024         | 0.952 ± 0.025     | 0.972 ± 0.022   | 0.961 ± 0.024 | 0.965 ± 0.023 | 0.962 ± 0.022   | 0.967 ± 0.025       | 0.968 ± 0.026                 | 0.957 ± 0.035   | 0.96 ± 0.032     | 0.967 ± 0.028     | 0.913 ± 0.034     | 0.93 ± 0.031          | 0.93 ± 0.036      | 0.943 ± 0.022     |
| hierarchical         | **1.0 ± 0.0**         | **1.0 ± 0.0**     | **1.0 ± 0.0**   | **1.0 ± 0.0** | **1.0 ± 0.0** | **1.0 ± 0.001** | **1.0 ± 0.0**       | **1.0 ± 0.0**                 | **1.0 ± 0.0**   | **1.0 ± 0.001**  | **0.999 ± 0.002** | **0.993 ± 0.014** | **0.991 ± 0.009**     | **0.983 ± 0.016** | **0.969 ± 0.035** |


![](images/sim_pre_clustering_AUC.png?raw=true)


## FBIRN Dataset 

| Clustering Algorithm | SVM               | Multilayer Perceptron | Logistic Regression | Passive Aggressive Classifier | Perceptron        | Random Forest     | Extra Trees       |
| -------------------- | ----------------- | --------------------- | ------------------- | ----------------------------- | ----------------- | ----------------- | ----------------- |
| kmeans               | *0.952 ± 0.036*   | *0.92 ± 0.065*        | *0.944 ± 0.039*     | *0.945 ± 0.035*               | **0.902 ± 0.043** | *0.871 ± 0.038*   | *0.853 ± 0.04*    |
| gmm                  | *0.936 ± 0.054*   | *0.946 ± 0.038*       | *0.943 ± 0.038*     | *0.929 ± 0.031*               | *0.882 ± 0.04*    | **0.885 ± 0.022** | **0.874 ± 0.026** |
| bgmm                 | *0.955 ± 0.037*   | *0.932 ± 0.042*       | *0.945 ± 0.038*     | *0.939 ± 0.038*               | *0.896 ± 0.074*   | *0.86 ± 0.039*    | *0.87 ± 0.056*    |
| dbscan               | 0.883 ± 0.027     | 0.893 ± 0.031         | 0.892 ± 0.033       | 0.884 ± 0.027                 | 0.828 ± 0.064     | 0.805 ± 0.064     | 0.806 ± 0.058     |
| hierarchical         | **0.957 ± 0.032** | **0.954 ± 0.038**     | **0.953 ± 0.038**   | **0.951 ± 0.032**             | *0.891 ± 0.098*   | *0.881 ± 0.032*   | *0.872 ± 0.048*   |


![](images/fbirn_pre_clustering_AUC.png?raw=true)


<img width="79%" src="results/fbirn_betas_accuracy.png?raw=True" />


<img width="20%" src="results/accuracy_legend.png?raw=true" />


## UCLA Dataset

| Clustering Algorithm | SVM              | Multilayer Perceptron | Logistic Regression | Passive Aggressive Classifier | Perceptron        | Extra Trees       | Random Forest     |
| -------------------- | ---------------- | --------------------- | ------------------- | ----------------------------- | ----------------- | ----------------- | ----------------- |
| kmeans               | *0.907 ± 0.057*  | *0.907 ± 0.057*       | *0.904 ± 0.06*      | **0.896 ± 0.08**              | *0.799 ± 0.116*   | *0.724 ± 0.168*   | *0.746 ± 0.133*   |
| gmm                  | **0.91 ± 0.059** | **0.909 ± 0.07**      | **0.908 ± 0.071**   | *0.885 ± 0.087*               | **0.886 ± 0.058** | *0.795 ± 0.095*   | *0.785 ± 0.108*   |
| bgmm                 | *0.909 ± 0.075*  | *0.907 ± 0.081*       | **0.908 ± 0.08**    | *0.877 ± 0.105*               | *0.879 ± 0.081*   | *0.741 ± 0.157*   | *0.705 ± 0.166*   |
| dbscan               | 0.409 ± 0.118    | 0.467 ± 0.131         | 0.69 ± 0.096        | 0.667 ± 0.122                 | 0.5 ± 0.0         | 0.643 ± 0.171     | 0.649 ± 0.125     |
| hierarchical         | *0.886 ± 0.054*  | *0.889 ± 0.07*        | *0.9 ± 0.069*       | *0.883 ± 0.071*               | *0.826 ± 0.122*   | **0.829 ± 0.099** | **0.792 ± 0.114** |


![](images/ucla_pre_clustering_AUC.png?raw=true)


<img width="79%" src="results/ucla_betas_accuracy.png?raw=True" />


<img width="20%" src="results/accuracy_legend.png?raw=true" />


### Initial UCLA Significant Comparisons

In order to evaluate the predictive capacities of the features produced by each clustering algorithm,
two-tailed t-test were performed to measure the statistical difference between the healthy control and the
schizophrenic patients.

The first t-test comparision was performed using the distribution of cluster assignments for each patient across
the time domain. For each time window, the average cluster assignment for all the healthy control patients was
compared to the average cluster assignment for all the schizophrenic patients. The corresponding p-values were then
tested at a significance level of 0.10.

The results displayed below highlight the points in time when there were less that a 10% chance that the observed
difference in a healthy control's cluster assignment and a schizophrenic's cluster assignment was due to normal
random variation. In theory, the more points of significance across time, the more likely a trained mode would be
able to correctly diagnose a subject. The results indicated that both k-means and gaussian mixture models failed to
produce statistically different cluster assignments over time. The Bayesian produced a few significant differences while
the hierarchical clustering algorithm was significant at every point in time. These results initially suggested that the
hierarchical clustering algorithm should outperform all the other clustering algorithms, but the subsequent results from
the trained supervised models refuted this hypothesis.

![](images/assignment_t_test_visualization.png?raw=true)

Given the lack of improvement in accuracy across all clustering algorithms, it was believed that training supervised
models using cluster assignment over time as input features was impracticable and would require much more data for
successful training. For each subject, there would be 130 time slots or features. To reduce the dimensionality of
the while maintaining relevant information, the beta coefficients were used in lieu of the time windows. The results
are displayed below.

![](images/beta_t_test_visualization.png?raw=true)

Each clustering algorithm found statistically different beta coefficients. The reduced feature space
(10 features for 267 patients for the UCLA data set) facilitated the correct classification of healthy and
schizophrenic patients across all clustering algorithms for all of the supervised learning algorithms.


# Section V: Conclusion

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
