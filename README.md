
# Voxel-wise encoding models for BIDS datasets with naturalistic stimuli
> This BIDS App lets you train voxelwise-encoding models for continuous (naturalistic) stimuli provided as a BIDS-compliant continuous recording file.


## Install

If you are only interested in using the Python module for preprocessing fMRI, lagging the stimulus, and training encoding models without the BIDS app, you can install this library by running `python setup.py` or `pip install voxelwiseencoding`.
You can use the BIDS app either via Docker or directly by calling `run.py`.

## Description

This app allows you to train voxel-wise encoding models for a BIDS dataset with a BIDS-compliant stimulus representation. To specify parameters for the processing of the stimulus (e.g.lagging and offsetting relative to fMRI), you can specify parameters that are supplied to `make_X_Y` in the `preprocessing` module as a JSON file. Similarly you can specify parameters to be supplied to `get_ridge_plus_scores` in the `encoding` module as a JSON as well.
Masking is done by default, by checking for masks in `output_dir/masks/` that are either named `sub-PARTICIPANT_LABEL_mask.nii.gz` (where PARTICIPANT_LABEL is the label provided by the user) or that are named `group_mask.nii.gz`. To disable masking call with the lag `--no-masking`.
Voxel-encoding models are trained in a cross-validation scheme: the parameter `n_splits` that is supplied to `get_ridge_plus_scores` via a configuation JSON file determines the number of folds in the cross-validation. Each fold is left out once, while a model is trained (and hyperparameters are tuned) on the remaining folds - model validation is done by voxel-wise [product moment correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between the predicted and observed fMRI activity for the left-out fold and saved as a 4D nifti in the output folder (with one image per left-out fold).
Similarly, for each left-out fold, Ridge regression models (trained on the remaining folds) are saved as a pickle file in the output folder.

### Example

We are going to use [this](https://openneuro.org/datasets/ds002322/versions/1.0.4) dataset to demonstrate an example workflow using the Python package.

First we need to download the data and extract a stimulus representation:


```python
!aws s3 sync --no-sign-request s3://openneuro.org/ds002322 ds002322-download/
import json
# these are the parameters for extracting a Mel spectrogram
# for computational ease in this example we want 1 sec segments of 31 Mel frequencies with a max frequency of * KHz
mel_params = {'n_mels': 31, 'sr': 16000, 'hop_length': 16000, 'n_fft': 16000, 'fmax': 8000}
with open('config.json', 'w+') as fl:
    json.dump(mel_params, fl)

!git clone https://github.com/mjboos/audio2bidsstim/
!pip install -r audio2bidsstim/requirements.txt
!python audio2bidsstim/wav_files_to_bids_tsv.py ds002322-download/stimuli/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav -c config.json
```

We then need to copy the extracted stimulus representation into the BIDS folder.

```python
!cp DownTheRabbitHoleFinal_mono_exp120_NR16_pad.tsv.gz ds002322-download/derivatives/task-alice_stim.tsv.gz
!cp DownTheRabbitHoleFinal_mono_exp120_NR16_pad.json ds002322-download/derivatives/sub-18/sub-18_task-alice_stim.json
```

And, lastly, because for this dataset the derivatives folder is missing timing information for the BOLD files - we are only interested in the TR - we have to copy that as well.

```python
!cp ds002322-download/sub-18/sub-18_task-alice_bold.json ds002322-download/derivatives/sub-18/sub-18_task-alice_bold.json 
```

We are now ready to define some model parameters and train the encoding model.

```python
from voxelwiseencoding.process_bids import run_model_for_subject

# these are the parameters used for preprocessing the BOLD fMRI files
bold_prep_params = {standardize: 'zscore', detrend: True}

# and for lagging the stimulus as well - we want to include 6 sec stimulus segments to predict fMRI
lagging_params = {'lag_time': 6}

# these are the parameters for sklearn's Ridge estimator
ridge_params = {'alphas': [1e-1, 1, 100, 1000], 'n_splits': 3, 'normalize': True}


ridges, scores, computed_mask = run_model_for_subject('18', 'ds002322-download/derivatives',
                                                      task='alice', mask='epi', bold_prep_kwargs=bold_prep_params,
                                                      preprocess_kwargs=lagging_params, encoding_kwargs=ridge_params)
```


## Documentation

[See here](https://mjboos.github.io/voxelwiseencoding) for further documentation about the Python package and consult the "Usage" section about the BIDS app/terminal usage.

## How to report errors

If you encounter errors with this code or have any questions about its uage, please open an issue on the Github repository [here](https://github.com/mjboos/voxelwiseencoding/).

## Usage

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
