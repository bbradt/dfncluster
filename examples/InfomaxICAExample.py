import numpy as np
from dfncluster.Preprocessing import PreprocessingSteps, GroupICA, NiftiPreprocessing

standard_pipeline = PreprocessingSteps(NiftiPreprocessing, GroupICA)
data = np.load('data/OpenNeuroDatasets/ds000030/Ds000030.npy')
standard_pipeline.apply(data)
