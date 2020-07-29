## Voxelwise Encoding BIDS App

This BIDS App lets you train voxelwise-encoding models for continuous (naturalistic) stimuli provided as a BIDS-compliant continuous recording file (see specification [here](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continuous-recordings.html)).

For more information about the specification of BIDS Apps see [here](https://docs.google.com/document/d/1E1Wi5ONvOVVnGhj21S1bmJJ4kyHFT7tkxnV3C23sjIE/).

### Description

TODO

### Documentation

TODO

### How to report errors

### Acknowledgments

### Usage

run.py /path/to/your/BIDS/dir /output/path --task your_task --ses session --skip_bids_validator --participant_label 01

This App has the following command line arguments:
Usage: run.py [-h]
              [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
              [--skip_bids_validator] [-d DESC] [-t TASK] [-s SES] [-v]
              [-r RECORDING] [--detrend] [--standardize STANDARDIZE]
              [--preprocessing-config PREPROCESSING_CONFIG]
              [--encoding-config ENCODING_CONFIG] [--identifier IDENTIFIER]
              [--log]
              bids_dir output_dir

Voxelwise Encoding BIDS App.

positional arguments:
  bids_dir              The directory with the input dataset formatted
                        according to the BIDS standard.
  output_dir            The directory where the output files should be stored.
                        If you want to mask the data please include folder
                        called masks that contains either subject-specific
                        NifTI masks named sub-<participant_label>_mask.nii.gz
                        or a group-level mask named group_mask.nii.gz.

optional arguments:
  -h, --help            show this help message and exit
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be
                        analyzed. The label corresponds to
                        sub-<participant_label> from the BIDS spec (so it does
                        not include "sub-"). If this parameter is not provided
                        all subjects should be analyzed. Multiple participants
                        can be specified with a space separated list.
  --skip_bids_validator
                        Whether or not to perform BIDS dataset validation
  -d DESC, --desc DESC  The label of the preprocessed data to use. Corresponds
                        to label in desc-<label> in the naming of the BOLD
                        NifTIs. If not provided, assumes no derivative label
                        is used.
  -t TASK, --task TASK  The task-label to use for training the voxel-wise
                        encoding model. Corresponds to label in task-<label>
                        in BIDS naming.
  -s SES, --ses SES     The label of the session to use. Corresponds to label
                        in ses-<label> in the BIDS directory.
  -v, --version         show program's version number and exit
  -r RECORDING, --recording RECORDING
                        The label of the stimulus recording to use.
                        Corresponds to label in recording-<label> of the
                        stimulus.
  --detrend             Whether to linearly detrend fMRI data voxel-wise
                        before training encoding models. Default is False.
  --standardize STANDARDIZE
                        How to voxel-wise standardize fMRI data before
                        training encoding models. Default is no
                        standardization, options are zscore for z-scoring and
                        psc for computing percent signal change.
  --preprocessing-config PREPROCESSING_CONFIG
                        Path to the preprocessing config file in JSON format.
                        Parameters in this file will be supplied as keyword
                        arguments to the make_X_Y function.
  --encoding-config ENCODING_CONFIG
                        Path to the encoding config file in JSON format.
                        Parameters in this file will be supplied as keyword
                        arguments to the get_ridge_plus_scores function.
  --identifier IDENTIFIER
                        Identifier to be included in the filenames for the
                        encoding model output.Use this to differentiate
                        different preprocessing steps or hyperparameters.
  --log                 Save preprocessing and model configuration together
                        with model output.

### Special considerations
