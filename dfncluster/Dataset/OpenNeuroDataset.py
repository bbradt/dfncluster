import os
import pandas as pd
import subprocess
import glob
import nibabel as nib
from nibabel.processing import resample_from_to
from dfncluster.Dataset import NiftiDataset
from dfncluster.Preprocessing.ICA.run_gift import gift_gica
import scipy.io as sio
from multiprocessing import Pool

AWS_COMMAND = 'aws s3 sync --no-sign-request s3://openneuro.org/%s %s'
MODALITIES = ['func']
SERIES = '*rest*.nii.gz'


def get_interpolated_nifti(template_filename, input_filename):
    '''
        Get an interpolated version of an file which is interpolated to match a reference.
        First, check if interpolated dimensions of nifti files match, if so, just return the input_filename.
        Else, if an interpolated version of the file has been created and saved in the root directory before, return its filename,
            else, create the interpolated version, and return its filename.

        Args:
            template_filename - the filename which has the desired spatial dimension
            input_filename - the filename to be interpolated

        Template for interpolated filenames example:
            input_filename = ' example.nii ' has dimension 53 x 63 x 52
            template_filename = 'template.nii' has dimension 53 x 63 x 46
            output_filename = 'example_INTERP_53_63_46.nii' has dimension 53 x 63 x 46
    '''

    base_dir = os.path.dirname(input_filename)
    input_prefix, input_ext = os.path.splitext(input_filename)
    template_img = nib.load(template_filename)
    input_img = nib.load(input_filename)
    template_img = template_img.slicer[:, :, :, :input_img.shape[3]]
    template_dim = template_img.shape

    if input_img.shape == template_dim:
        return input_filename

    output_filename = os.path.join(
        base_dir, "%s_INTERP_%d_%d_%d.nii" % (input_prefix, template_img.shape[0], template_img.shape[1], template_img.shape[2]))

    if os.path.exists(output_filename):
        return output_filename

    output_img = resample_from_to(input_img, template_img)
    nib.save(output_img, output_filename)

    return output_filename


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
        participants = pd.read_csv(os.path.join(directory, study_name, partfile), sep='\t')
        subjects = []
        for index, row in participants.iterrows():
            participant = row['participant_id']
            subject_dir = os.path.join(directory, study_name, participant)
            for modality in modalities:
                modality_dir = os.path.join(subject_dir, modality)
                file_search = glob.glob(os.path.join(modality_dir, series))
                for filename in file_search:
                    series_name, _ = os.path.splitext(
                        os.path.basename(filename))
                    series_name = series_name.replace(participant, '').replace('.nii', '').replace('.gz', '')[1:]
                    template = get_interpolated_nifti(os.path.abspath(filename), os.path.abspath('NeuroMark.nii'))
                    new_dir = '/data/mialab/users/bbaker/projects/dfncluster/data/MatDatasets/ds000115/subjects_new'
                    new_savename = os.path.join(new_dir, 'subject_%d_%s.mat' % (index, series_name))
                    if not os.path.exists(new_savename):
                        result = gift_gica([os.path.abspath(filename)], algoType=16, refFiles=os.path.abspath(template), run_name="subject_%s_%s" % (index, modality),
                                           out_dir=new_dir)
                        new_file = os.path.join(new_dir, 'gica_cmd_ica_br1.mat')
                        ica_result = sio.loadmat(new_file)
                        TC = ica_result['compSet']['tc'].item().T
                        sio.savemat(new_savename, {'TC': TC})
                    row[series_name] = new_savename
                    print('GIGICA - - Created new file %s' % new_savename)
            subjects.append(row.to_dict())
        df = pd.DataFrame(subjects)
        df.to_csv(os.path.join(directory, 'data.csv'), index=False)
        kwargs['filename'] = os.path.join(directory, 'data.csv')
        x, y = super(OpenNeuroDataset, self).generate(**kwargs)
        return x, y
