import os
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

