import os
import numpy as np
from dfncluster.Preprocessing import PreprocessingSteps, GroupICA, NiftiPreprocessing
from dfncluster.Dataset import OpenNeuroDataset
import scipy.io as sio

print("Instantiating pipeline")
standard_pipeline = PreprocessingSteps(NiftiPreprocessing(), GroupICA())
print("Loading data")
#data = OpenNeuroDataset.load('data/OpenNeuroDatasets/ds000030/ds000030_subset.npy', large=False)
dataset = OpenNeuroDataset("ds000030",
                           directory='data/OpenNeuroDatasets/ds000030',
                           feature_columns=['task-rest_bold'],
                           label_columns=['diagnosis'],
                           subset_size=0.30,
                           pre_shuffle=True)
print("Applying pipeline")
S = standard_pipeline.apply(dataset.features)
ds = 'results/ica/ds000030/derivatives'
result_file = os.path.join(ds, 'data.csv')
os.makedirs(ds, exist_ok=True)
with open(result_file, 'w') as file:
    file.write('subject_id,diagnosis,filename\n')
for i, (subject, label) in enumerate(zip(S, dataset.labels)):
    sid = 'subject_%d' % i
    sfn = os.path.join(ds, sid+'.mat')
    sio.savemat(sfn, {"TC": subject})
    with open(result_file, 'a') as file:
        file.write('%s,%s,%s\n' % (sid, int(label[0] == 'SCHZ'), sfn))
