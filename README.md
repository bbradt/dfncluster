# Introduction

Schizophrenia is a chronic and serious mental disorder which affects how a person thinks, feels, and behaves. Although there have been many studies about psychological and behavioral manifestations of schizophrenia, neuroscientists have yet to determine a set of corresponding neurological biomarkers for this disorder. A functional magnetic resonance imaging (fMRI) can help determine non-invasive biomarkers for schizophrenia in brain function[[1]](#ref1)[[2]](#ref2) and one such fMRI analysis technique called, dynamic functional network connectivity (dFNC)[[3]](#ref3)[[4]](#ref4)[[10]](#ref10), uses K-Means clustering to characterize time-varying connectivity between functional networks. Researchers have worked on finding correlation between schizophrenia and dFNC[[1]](#ref1)[[2]](#ref2)[[5]](#ref5)[[6]](#ref6), but little work has been done with the choice of clustering algorithm[[7]](#ref7)[[9]](#ref9). Therefore, in this project, we have studied how modifying the clustering technique in the dFNC pipeline can yield dynamic states from fMRI data that impact the accuracy of classifying schizophrenia[[8]](#ref8).

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

## Initial UCLA Significant Comparisons

In order to evaluate the predictive capacities of the features produced by each clustering algorithm, a
two-tailed t-test was performed comparing the healthy control and the schizophrenic patients.

The first t-test comparision was performed using the cluster assignments for each patient across
the time domain where each time slot represented a feature of the training data. For each time slot (feature),
the average cluster assignment for all the healthy control patients was compared to the average cluster assignment
for all the schizophrenic patients. The corresponding p-values were then tested at a significance level of 0.10.

The results displayed below highlight the points in time when there was less that a 10% chance that the
observed difference in a healthy control's cluster assignment and a schizophrenic's cluster assignment was due to normal
random variation. In theory, the more points of significance across time the more likely a trained model would accurately
diagnose a subject. The results indicated that both K-Means and Gaussian Mixture Models failed to produce statistically
different cluster assignments across time. The Bayesian Gaussian Mixture Model produced some significant differences while
the Hierarchical clustering was significant at every time point.

These results initially suggested that Hierarchical clustering would outperform all the other clustering algorithms,
but the subsequent testing disporved this hypothesis.

![](images/assignment_t_test_visualization.png?raw=true)

Given the lack of improvement in accuracy across all clustering algorithms, it was believed that training supervised
models using using time points as features would require much more data for successful classification.
Using time slots as features meant that there were 130 features in the data. Since there were only 267 patients
in the UCLA data set, it was surmised that the dimensionality of the data was too high. To reduce the dimensionality
the beta coefficients were calculated for each subject reducing the number of training features form 130 to 10.

![](images/beta_t_test_visualization.png?raw=true)

Each clustering algorithm found statistically different beta coefficients. The reduced feature space facilitated the
correct classification of healthy and schizophrenic patients regardless of the clustering algorithm used for each
supervised learning model.
## Gaussian Simulated Dataset 

| Clustering Algorithm | Multilayer Perceptron | Nearest Neighbors | SVM             | Random Forest | Extra Trees   | Gradient Boost  | Logistic Regression | Passive Aggressive Classifier | Perceptron      | Gaussian Process | Ada Boost         | Voting            | Bernoulli Naive Bayes | Bagging           | Decision Tree     |
| -------------------- | --------------------- | ----------------- | --------------- | ------------- | ------------- | --------------- | ------------------- | ----------------------------- | --------------- | ---------------- | ----------------- | ----------------- | --------------------- | ----------------- | ----------------- |
| kmeans               | 0.972 ± 0.026         | 0.96 ± 0.021      | 0.972 ± 0.023   | 0.962 ± 0.027 | 0.966 ± 0.024 | 0.954 ± 0.031   | 0.971 ± 0.022       | 0.974 ± 0.02                  | *0.947 ± 0.062* | 0.955 ± 0.027    | 0.957 ± 0.032     | 0.92 ± 0.037      | 0.948 ± 0.022         | 0.941 ± 0.035     | *0.938 ± 0.036*   |
| gmm                  | 0.963 ± 0.028         | 0.954 ± 0.03      | 0.972 ± 0.025   | 0.967 ± 0.022 | 0.976 ± 0.017 | 0.928 ± 0.046   | 0.962 ± 0.028       | 0.97 ± 0.024                  | 0.962 ± 0.028   | 0.952 ± 0.029    | 0.955 ± 0.031     | 0.92 ± 0.045      | 0.938 ± 0.028         | 0.917 ± 0.036     | 0.923 ± 0.033     |
| bgmm                 | 0.966 ± 0.023         | 0.949 ± 0.028     | *0.974 ± 0.027* | 0.966 ± 0.026 | 0.963 ± 0.028 | 0.962 ± 0.03    | 0.974 ± 0.02        | 0.972 ± 0.024                 | 0.969 ± 0.029   | 0.955 ± 0.029    | 0.958 ± 0.029     | 0.923 ± 0.045     | 0.949 ± 0.028         | 0.931 ± 0.043     | 0.933 ± 0.026     |
| dbscan               | 0.971 ± 0.024         | 0.952 ± 0.025     | 0.972 ± 0.022   | 0.961 ± 0.024 | 0.965 ± 0.023 | 0.962 ± 0.022   | 0.967 ± 0.025       | 0.968 ± 0.026                 | 0.957 ± 0.035   | 0.96 ± 0.032     | 0.967 ± 0.028     | 0.913 ± 0.034     | 0.93 ± 0.031          | 0.93 ± 0.036      | 0.943 ± 0.022     |
| hierarchical         | **1.0 ± 0.0**         | **1.0 ± 0.0**     | **1.0 ± 0.0**   | **1.0 ± 0.0** | **1.0 ± 0.0** | **1.0 ± 0.001** | **1.0 ± 0.0**       | **1.0 ± 0.0**                 | **1.0 ± 0.0**   | **1.0 ± 0.001**  | **0.999 ± 0.002** | **0.993 ± 0.014** | **0.991 ± 0.009**     | **0.983 ± 0.016** | **0.969 ± 0.035** |


![](images/sim_pre_clustering_AUC.png?raw=true)

Above plot shows "Area Under Curve" of various classifiers on simulated Gaussian Data before clustering where SVM seems to be performing the best followed by "Ada Boost" and "Gradient Boost" while "Bernoulli Naive Bayes" seems to be performing the worst.

<img width="79%" src="results/gauss_betas_accuracy.png?raw=True" />

<img width="20%" src="results/accuracy_legend.png?raw=true" />

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data using beta features. Accuracy has improved a lot in comparison to the previous case above without clusterer and without using beta feature say for example from 0.4 to 0.6 for KMeans and likewise for other clusterer. Multilayer percepteon classifier shows some improvement in GMM over KMeans and Passive Aggressive Classifier in BGMM shows some improvement over KMeans clustering.

## FBIRN Dataset 

| Clustering Algorithm | SVM               | Multilayer Perceptron | Logistic Regression | Passive Aggressive Classifier | Perceptron        | Random Forest     | Extra Trees       |
| -------------------- | ----------------- | --------------------- | ------------------- | ----------------------------- | ----------------- | ----------------- | ----------------- |
| kmeans               | *0.952 ± 0.036*   | *0.92 ± 0.065*        | *0.944 ± 0.039*     | *0.945 ± 0.035*               | **0.902 ± 0.043** | *0.871 ± 0.038*   | *0.853 ± 0.04*    |
| gmm                  | *0.936 ± 0.054*   | *0.946 ± 0.038*       | *0.943 ± 0.038*     | *0.929 ± 0.031*               | *0.882 ± 0.04*    | **0.885 ± 0.022** | **0.874 ± 0.026** |
| bgmm                 | *0.955 ± 0.037*   | *0.932 ± 0.042*       | *0.945 ± 0.038*     | *0.939 ± 0.038*               | *0.896 ± 0.074*   | *0.86 ± 0.039*    | *0.87 ± 0.056*    |
| dbscan               | 0.883 ± 0.027     | 0.893 ± 0.031         | 0.892 ± 0.033       | 0.884 ± 0.027                 | 0.828 ± 0.064     | 0.805 ± 0.064     | 0.806 ± 0.058     |
| hierarchical         | **0.957 ± 0.032** | **0.954 ± 0.038**     | **0.953 ± 0.038**   | **0.951 ± 0.032**             | *0.891 ± 0.098*   | *0.881 ± 0.032*   | *0.872 ± 0.048*   |


![](images/fbirn_pre_clustering_AUC.png?raw=true)

Above plot shows "Area Under Curve" of various classifiers on FBirn Data before clustering where "Random Forest" seems to be performing the best followed by "Gradient Boost" and "Multi Layer Perceptron" while "Decision Tree" and "Bernoulli Naive Bayes" seems to be performing the worst.

![](results/fbirn_assignments_accuracy.png?raw=true)

<img width="20%" src="results/accuracy_legend.png?raw=true" />

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data without using beta features. Accuracy has improved a lot in comparison to the previous case above without clusterer and without using beta feature say for example from 0.5 to 0.7 for KMeans with "Multi Layer Perceptron" and likewise for other clusterer. Multilayer percepteon classifier shows some improvement in GMM over KMeans and "Logistic Regression" and "Passive Aggressive Classifier" in BGMM shows some improvement over KMeans clustering.

<img width="79%" src="results/fbirn_betas_accuracy.png?raw=True" />

<img width="20%" src="results/accuracy_legend.png?raw=true" />

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data using beta features. Accuracy has improved a lot in comparison to the previous case above with clusterer but without using beta feature say for example from 0.7 to 0.9 for KMeans with "Multi Layer Perceptron" and likewise for other clusterer. "Random Forest" and "Extra Trees" classifier shows some improvement in GMM over KMeans and "Random Forest", "Extra Trees"  and "Perceptron" Classifier in BGMM shows some improvement over KMeans clustering.

## UCLA Dataset

| Clustering Algorithm | SVM              | Multilayer Perceptron | Logistic Regression | Passive Aggressive Classifier | Perceptron        | Extra Trees       | Random Forest     |
| -------------------- | ---------------- | --------------------- | ------------------- | ----------------------------- | ----------------- | ----------------- | ----------------- |
| kmeans               | *0.907 ± 0.057*  | *0.907 ± 0.057*       | *0.904 ± 0.06*      | **0.896 ± 0.08**              | *0.799 ± 0.116*   | *0.724 ± 0.168*   | *0.746 ± 0.133*   |
| gmm                  | **0.91 ± 0.059** | **0.909 ± 0.07**      | **0.908 ± 0.071**   | *0.885 ± 0.087*               | **0.886 ± 0.058** | *0.795 ± 0.095*   | *0.785 ± 0.108*   |
| bgmm                 | *0.909 ± 0.075*  | *0.907 ± 0.081*       | **0.908 ± 0.08**    | *0.877 ± 0.105*               | *0.879 ± 0.081*   | *0.741 ± 0.157*   | *0.705 ± 0.166*   |
| dbscan               | 0.409 ± 0.118    | 0.467 ± 0.131         | 0.69 ± 0.096        | 0.667 ± 0.122                 | 0.5 ± 0.0         | 0.643 ± 0.171     | 0.649 ± 0.125     |
| hierarchical         | *0.886 ± 0.054*  | *0.889 ± 0.07*        | *0.9 ± 0.069*       | *0.883 ± 0.071*               | *0.826 ± 0.122*   | **0.829 ± 0.099** | **0.792 ± 0.114** |


![](images/ucla_pre_clustering_AUC.png?raw=true)

Above plot shows "Area Under Curve" of various classifiers on UCLA Data before clustering where SVM seems to be performing the best followed by "Multilayer Perceptron" and "Gradient Boost" while "Gaussian Process" and "Decision Tree" seems to be performing the worst.

<img width="79%" src="results/ucla_betas_accuracy.png?raw=True" />

<img width="20%" src="results/accuracy_legend.png?raw=true" />

Above plot shows "Accuracy" of various classifiers such as KMeans, Gaussian Mixture Model(GMM), Bayesian Gaussian Mixture Model(BGMM), Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and Hierarchical clustering methods on Simulated Gaussian Data using beta features. Accuracy has improved a lot in comparison to the previous case above without clusterer and without using beta feature say for example from 0.7 to 0.9 for KMeans with "Multi Layer Perceptron" and likewise for other clusterer. Almost all the classifiers in all the clusterer shows improvement over KMeans clustering.

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
