import os
import warnings
import numpy as np
import joblib
from nilearn.masking import unmask, apply_mask
from nibabel import save, load
from nilearn.signal import clean


def preprocess_bold_fmri(bold, mask=None, detrend=True, standardize='zscore', **kwargs):
    '''Preprocesses data and returns ndarray.'''
    if mask:
        data = apply_mask(bold, mask)
    else:
        data = load(bold).get_data()
        data = np.reshape(data, (-1, data.shape[-1])).T
    return clean(data, detrend=detrend, standardize=standardize, **kwargs)

def get_remove_idx(lagged_stimulus, remove_nan=True):
    '''Returns indices of rows in lagged_stimulus to remove'''
    if remove_nan is True:
        return np.where(np.any(np.isnan(lagged_stimulus), axis=1))[0]
    elif remove_nan <= 1. and remove_nan >= 0.:
        return np.where(np.isnan(lagged_stimulus).mean(axis=1) > remove_nan)[0]
    else:
        raise ValueError('remove_nan needs to be either True, False, or a float between 0 and 1.')

def make_X_Y(stimuli, fmri, lag_time=6.0, start_times=None, offset_stim=2., TR=2., stim_TR=0.1, filler_value=np.nan, remove_nans=0.5):
    '''Creates (lagged) features and fMRI matrices concatenated along runs
    
    INPUT:
    stimuli      - list of stimulus representations
    fmri         - list of fMRI ndarrays
    lag_time     - lag to introduce for stimuli in seconds
    start_times  - starting time of the stimuli relative to fMRI recordings in seconds
                   appends zeros to stimulus representation to match fMRI and stimulus
    offset_stim  - time to offset stimulus relative to fMRI in the lagged stimulus,
                   i.e. when predicting fmri at time t use only stimulus features
                   before t-offset_stim. This reduces the number of time points used
                   in the model.
    TR           - repetition time of the fMRI data in seconds
    stim_TR      - repetition time of the stimulus in seconds
    filler_value - appends filler_value to stimulus array to account for starting_time
                   use np.nan here with remove_nans=True to remove fmri/stimulus samples where no stimulus was presented
    remove_nans  - bool or float 0<=remove_nans<=1,
                   True/False indicate whether to remove all or none
                   stimulus/fmri samples that contain nans
                   a proportion keeps all samples in the lagged stimulus that have
                   lower number of nans than this proportion.
                   Replace nans with zeros in this case.
    '''
    from skimage.util import view_as_windows
    if len(stimuli) != len(fmri):
        raise ValueError('Stimulus and fMRI need to have the same number of runs. '
        'Instead fMRI has {} and stimulus {} runs.'.format(len(fmri), len(stimuli)))
    # find out temporal alignment
    stim_samples_per_TR = TR / stim_TR
    # check if result is close to an integer
    if not np.isclose(stim_samples_per_TR, np.round(stim_samples_per_TR)):
        warnings.warn('Stimulus timing and fMRI timing do not align. '
        'Stimulus samples per fMRI samples: {0} for stimulus TR {1} and fMRI TR {2}. '
        'Proceeds by rounding stimulus samples '
        'per TR.'.format(stim_samples_per_TR, stim_TR, TR), RuntimeWarning)
    stim_samples_per_TR = int(np.round(stim_samples_per_TR))
    # check if lag time is multiple of TR
    if not (np.isclose(lag_time % TR, 0) or np.isclose(lag_time % TR, np.round(TR))):
        raise ValueError('lag_time should be a multiple of TR so '
                'that stimulus/fMRI alignment does not change.')

    n_features = stimuli[0].shape[1]
    if not np.all(np.array([stim.shape[1] for stim in stimuli]) == n_features):
        raise ValueError('Stimulus has different number of features per run.')

    lag_TR = int(np.round(lag_time / TR))
    # append filler values before lagging
    for i in range(len(stimuli)):
        n_append = 0
        n_prepend = 0
        # check if the stimulus start time is moved w.r.t. fmri
        if start_times:
            n_prepend += int(np.round(start_times[i] / stim_TR))
            stimuli[i] = np.vstack(
                    [np.full((n_prepend, n_features), filler_value),
                     stimuli[i]])
        # make reshapeable by appending filler
        if (stimuli[i].shape[0]) % stim_samples_per_TR > 0:
            # either remove part of the stimulus (if it is longer than fmri) or append filler
            if stimuli[i].shape[0] / stim_samples_per_TR > fmri[i].shape[0]:
                stimuli[i] = stimuli[i][:-(stimuli[i].shape[0] % stim_samples_per_TR)]
            else:
                n_append = stim_samples_per_TR - ((stimuli[i].shape[0]) % stim_samples_per_TR)
                stimuli[i] = np.vstack([
                        stimuli[i],
                        np.full((n_append, n_features), filler_value)])
        # now reshape and lag
        # TODO: check for memory footprint wrt copying
        stimuli[i] = np.reshape(stimuli[i], (-1, stim_samples_per_TR * n_features))
        # account for lagging
        n_prepend_lag = lag_TR - 1
        stimuli[i] = np.vstack(
                [np.full((n_prepend_lag, n_features * stim_samples_per_TR), filler_value),
                 stimuli[i]])

        stimuli[i] = np.squeeze(view_as_windows(stimuli[i], (lag_TR, 1)))
        stimuli[i] = np.reshape(stimuli[i], (stimuli[i].shape[0], -1))

    # remove nans in stim/fmri here
    if remove_nans:
        for i in range(len(stimuli)):
            remove_idx = get_remove_idx(stimuli[i], remove_nans)
            stimuli[i] = np.delete(stimuli[i], remove_idx, axis=0)
            fmri[i] = np.delete(fmri[i], remove_idx, axis=0)

    # remove stimulus representations that are more recent than offset_stim
    if offset_stim > 0:
        for i in range(len(stimuli)):
            stimuli[i] = stimuli[i][:, :-int(np.round(offset_stim / stim_TR))]

    # remove fmri samples recorded after stimulus has ended
    for i in range(len(stimuli)):
        if fmri[i].shape[0] != stimuli[i].shape[0]:
            warnings.warn('fMRI data and stimulus samples differ. '
            'Removing additional fMRI samples. This could mean that you recorded '
            'long after stimulus ended or that something went wrong in the '
            'preprocessing. fMRI: {}s stimulus: {}s'.format(
                TR*fmri[i].shape[0], TR*stimuli[i].shape[0]), RuntimeWarning)
            if fmri[i].shape[0] > stimuli[i].shape[0]:
                fmri[i] = fmri[i][:-(fmri[i].shape[0]-stimuli[i].shape[0])]

    return np.vstack(stimuli), np.vstack(fmri)
