#!/usr/bin/env python3
import argparse
import os
import subprocess
import nibabel
import numpy
from glob import glob
from preprocessing import preprocess_bold_fmri, make_X_Y
import joblib
import json
import numpy as np

__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()

# FIXME: temporary for dev
memory = joblib.Memory('/data/mboos/joblib')
preprocess_bold_fmri = memory.cache(preprocess_bold_fmri)


def create_stim_filename_from_args(subject_label, **kwargs):
    '''Creates an expression corresponding to the stimulus files. It does not differentiate between json and tsv(.gz) files yet.'''
    stim_expr = ['sub-{}'.format(subject_label),
                 'ses-{}'.format(kwargs['ses']) if kwargs['ses'] else None,
                 'task-{}'.format(kwargs['task']) if kwargs['task'] else None,
                 'desc-{}'.format(kwargs['desc']) if kwargs['desc'] else None,
                 '*',
                 'recording-{}'.format(kwargs['recording']) if kwargs['recording'] else None,
                 'stim']
    stim_expr = '_'.join([term for term in stim_expr if term])
    return stim_expr



def get_func_bold_directory(subject_label, **kwargs):
    '''Returns a path to the directory in which the bold files of the given subject reside'''
    bold_folder = [kwargs['bids_dir'], 'sub-{}'.format(subject_label),
                 'ses-{}'.format(kwargs['ses']) if kwargs['ses'] else None,
                 'func']
    bold_folder = os.path.join(*[term for term in bold_folder if term])
    return bold_folder


def create_bold_glob_from_args(subject_label, **kwargs):
    '''Creates a globbable expression corresponding to the bold NifTIs to be used.'''
    bold_expr = ['sub-{}'.format(subject_label),
                 'ses-{}'.format(kwargs['ses']) if kwargs['ses'] else None,
                 'task-{}'.format(kwargs['task']) if kwargs['task'] else None,
                 'desc-{}'.format(kwargs['desc']) if kwargs['desc'] else None,
                 '*_bold.nii.gz']
    bold_expr = '_'.join([term for term in bold_expr if term])
    return bold_expr


def run(command, env={}):
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d"%process.returncode)

parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you want to mask the data please include '
                    'folder called masks that contains either subject-specific NifTI '
                    'masks named sub-<participant_label>_mask.nii.gz or a group-level '
                    'mask named group_mask.nii.gz.')
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-"). If this parameter is not '
                   'provided all subjects should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+")
parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                   action='store_true')
parser.add_argument('-d', '--desc', help='The label of the preprocessed '
                    'data to use. Corresponds to label in desc-<label> in the '
                    'naming of the BOLD NifTIs. If not provided, assumes no derivative '
                    'label is used.')
parser.add_argument('-t', '--task', help='The task-label to use for training the voxel-wise encoding model. Corresponds to label in task-<label> in BIDS naming.')
parser.add_argument('-s', '--ses', help='The label of the session to use. '
                    'Corresponds to label in ses-<label> in the BIDS directory.')
parser.add_argument('-v', '--version', action='version',
                    version='BIDS-App example version {}'.format(__version__))
parser.add_argument('-r', '--recording', help='The label of the stimulus recording to use. '
                    'Corresponds to label in recording-<label> of the stimulus.')

# TODO: make list of arguments to be included

# split
# what to save


args = parser.parse_args()

if not args.skip_bids_validator:
    run('bids-validator %s'%args.bids_dir)

subjects_to_analyze = []
# only for a subset of subjects
if args.participant_label:
    subjects_to_analyze = args.participant_label
# for all subjects
else:
    subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

with open(os.path.join(args.bids_dir, 'task-{}_bold.json'.format(args.task)), 'r') as fl:
    task_meta = json.load(fl)

for subject_label in subjects_to_analyze:
    bold_folder = get_func_bold_directory(subject_label, **vars(args))
    bold_glob = create_bold_glob_from_args(subject_label, **vars(args))
    bold_files = sorted(glob(os.path.join(bold_folder, bold_glob)))
    stim_glob = create_stim_filename_from_args(subject_label, **vars(args))

    # first check if there exist subject specific stimulus files
    stim_tsv = glob(os.path.join(bold_folder, '.'.join([stim_glob, 'tsv.gz'])))
    if not stim_tsv:
        # try to get uncompressed tsv
        stim_tsv = glob(os.path.join(bold_folder, '.'.join([stim_glob, 'tsv'])))
        if not stim_tsv:
            # try to get tsvs in root directory without subject specifier
            root_glob = '_'.join(stim_glob.split('_')[1:])
            stim_tsv = glob(os.path.join(args.bids_dir,
                                              '.'.join([root_glob, 'tsv.gz'])))
            if not stim_tsv:
                # and check again in root for tsv
                stim_tsv = glob(os.path.join(args.bids_dir,
                                                  '.'.join([root_glob, 'tsv'])))
                if not stim_tsv:
                    raise ValueError('No stimulus files found! [Mention naming scheme and location here]')
    stim_tsv = sorted(stim_tsv)
    stim_json = sorted(glob(os.path.join(bold_folder, '.'.join([stim_glob, 'json']))))
    if not stim_json:
        raise ValueError('No stimulus json files found! [mention more]')

    if not (len(stim_tsv) == len(stim_json) and len(stim_json) == len(bold_files)):
        raise ValueError('Number of stimulus tsv, stimulus json, and BOLD files differ.'
                ' Stimulus json: {} \n stimulus tsv: {} \n BOLD: {}'.format(stim_json, stim_tsv, bold_files))

    mask = None
    masks_path = os.path.join(args.output_dir, 'masks')
    if os.path.exists(masks_path):
        if os.path.exists(os.path.join(masks_path, 'sub-{}_mask.nii.gz'.format(subject_label))):
            mask = os.path.join(masks_path, 'sub-{}_mask.nii.gz'.format(subject_label))
        elif os.path.exists(os.path.join(masks_path, 'group_mask.nii.gz')):
            mask = os.path.join(masks_path, 'group_mask.nii.gz')

    # do BOLD preprocessing
    preprocessed_data = []
    for bold_file in bold_files:
        preprocessed_data.append(preprocess_bold_fmri(bold_file, mask=mask))

    # load stimulus
    stim_meta = []
    stimuli = []
    for tsv_fl, json_fl in zip(stim_tsv, stim_json):
        with open(json_fl, 'r') as fl:
            stim_meta.append(json.load(fl))
        stimuli.append(np.loadtxt(tsv_fl, delimiter='\t'))

    start_times = [st_meta['StartTime'] for st_meta in stim_meta]
    stim_TR = 1 / stim_meta[0]['SamplingFrequency']
    # add parameters from args
    stimuli, preprocessed_data = make_X_Y(stimuli, preprocessed_data, start_times=start_times, TR=task_meta['RepetitionTime'], stim_TR=stim_TR)


