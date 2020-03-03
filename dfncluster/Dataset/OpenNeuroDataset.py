import os
import subprocess

from dfncluster.Dataset import NiftiDataset

AWS_COMMAND = 'aws s3 sync --no-sign-request s3://openneuro.org/%s %s'

class OpenNeuroDataset(NiftiDataset):
    def __init__(self, study_name, directory='.'):
        subprocess.call(AWS_COMMAND % (study_name, os.path.join(directory, study_name)), shell=True)
        
