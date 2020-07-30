![Python package](https://github.com/mjboos/voxelwiseencoding/workflows/Python%20package/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Voxelwise Encoding BIDS App

This BIDS App lets you train voxelwise-encoding models for continuous (naturalistic) stimuli provided as a BIDS-compliant continuous recording file (see specification [here](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continuous-recordings.html)).
For more information about the specification of BIDS Apps see [here](https://docs.google.com/document/d/1E1Wi5ONvOVVnGhj21S1bmJJ4kyHFT7tkxnV3C23sjIE/).
For auditory stimuli [this](https://github.com/mjboos/audio2bidsstim/) module can help you convert your wav file to a BIDS stimulus representation.
If you are only interested in using the Python module for preprocessing fMRI, lagging the stimulus, and training encoding models without the BIDS app, you can install this library by running `python setup.py` or using pip to install it.

### Description

This app allows you to train voxel-wise encoding models for a BIDS dataset with a BIDS-compliant stimulus representation. To specify parameters for the processing of the stimulus (e.g.lagging and offsetting relative to fMRI), you can specify parameters that are supplied to `make_X_Y` in the `preprocessing` module as a JSON file. Similarly you can specify parameters to be supplied to `get_ridge_plus_scores` in the `encoding` module as a JSON as well.
Masking is done by default, by checking for masks in `output_dir/masks/` that are either named `sub-PARTICIPANT_LABEL_mask.nii.gz` (where PARTICIPANT_LABEL is the label provided by the user) or that are named `group_mask.nii.gz`. To disable masking call with the lag `--no-masking`.

### Documentation

For further documentation consult the "Usage" section or the documentation in `voxelwiseencoding/preprocessing.py` and `voxelwiseencoding/encoding.py`.

### How to report errors

If you encounter errors with this code or have any questions about its uage, please open an issue on the Github repository [here](https://github.com/mjboos/voxelwiseencoding/).

### Usage

run.py /path/to/your/BIDS/dir /output/path --task your_task --ses session --skip_bids_validator --participant_label 01

This App has the following command line arguments:
<pre>usage: run.py [-h]
              [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
              [--skip_bids_validator] [-d DESC] [-t TASK] [-s SES] [-v]
              [-r RECORDING] [--detrend] [--standardize STANDARDIZE]
              [--preprocessing-config PREPROCESSING_CONFIG]
              [--encoding-config ENCODING_CONFIG] [--identifier IDENTIFIER]
              [--no-masking] [--log]
              bids_dir output_dir

Voxelwise Encoding BIDS App.

positional arguments:
  bids_dir              The directory with the input dataset formatted
                        according to the BIDS standard.
  output_dir            The directory where the output files should be stored.
                        If you want to mask the data please include folder
                        called masks that contains either subject-specific
                        NifTI masks named sub-&lt;participant_label&gt;_mask.nii.gz
                        or a group-level mask named group_mask.nii.gz.

optional arguments:
  -h, --help            show this help message and exit
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be
                        analyzed. The label corresponds to
                        sub-&lt;participant_label&gt; from the BIDS spec (so it does
                        not include &quot;sub-&quot;). If this parameter is not provided
                        all subjects should be analyzed. Multiple participants
                        can be specified with a space separated list.
  --skip_bids_validator
                        Whether or not to perform BIDS dataset validation
  -d DESC, --desc DESC  The label of the preprocessed data to use. Corresponds
                        to label in desc-&lt;label&gt; in the naming of the BOLD
                        NifTIs. If not provided, assumes no derivative label
                        is used.
  -t TASK, --task TASK  The task-label to use for training the voxel-wise
                        encoding model. Corresponds to label in task-&lt;label&gt;
                        in BIDS naming.
  -s SES, --ses SES     The label of the session to use. Corresponds to label
                        in ses-&lt;label&gt; in the BIDS directory.
  -v, --version         show program&apos;s version number and exit
  -r RECORDING, --recording RECORDING
                        The label of the stimulus recording to use.
                        Corresponds to label in recording-&lt;label&gt; of the
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
  --no-masking          Flag to disable masking. This will lead to many non-
                        brain voxels being included.
  --log                 Save preprocessing and model configuration together
                        with model output.
</pre>
### Special considerations
