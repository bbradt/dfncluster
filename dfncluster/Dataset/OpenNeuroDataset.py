import os
import pandas as pd
import subprocess
import glob
from dfncluster.Dataset import NiftiDataset

AWS_COMMAND = 'aws s3 sync --no-sign-request s3://openneuro.org/%s %s'
MODALITIES = ['func']
SERIES = '*rest*.nii.gz'


class OpenNeuroDataset(NiftiDataset):
    def __init__(self, study_name='', directory='.', partfile='participants.tsv', modalities=MODALITIES, series=SERIES, **kwargs):
        super(OpenNeuroDataset, self).__init__(study_name=study_name,
                                               directory=directory,
                                               partfile=partfile,
                                               series=series,
                                               modalities=modalities,
                                               **kwargs)

    def generate(self, **kwargs):
        study_name = kwargs['study_name']
        directory = kwargs['directory']
        partfile = kwargs['partfile']
        modalities = kwargs['modalities']
        series = kwargs['series']
        if not os.path.exists(os.path.join(directory, study_name)):
            subprocess.call(AWS_COMMAND % (study_name,
                                           os.path.join(directory, study_name)),
                            shell=True)
        participants = pd.read_csv(os.path.join(directory, partfile))
        subjects = []
        for index, row in participants.iterrows():
            participant = row['participant_id']
            subject_dir = os.path.join(directory, participant)
            for modality in modalities:
                modality_dir = os.path.join(subject_dir, modality)
                file_search = glob.glob(os.path.join(modality_dir, series))
                for filename in file_search:
                    series_name, _ = os.path.splitext(
                        os.path.basename(filename))
                    row[series_name] = filename
