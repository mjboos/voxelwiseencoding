#change to better import
from voxelwiseencoding import preprocessing as prep
import nibabel
import numpy as np

def create_test_data():
    '''Creates toy Nifti images and data to test voxelwise encoding models.'''
    # reshape to simulate different temporal resolution of stim and fmri
    X = np.reshape(np.random.randn(1000, 5), (100, -1))
    betas = np.random.randn(X.shape[-1], 27) * 2
    y = X.dot(betas).T
    mask = np.ones((3, 3, 3), np.bool)
    mask_img = nibabel.Nifti1Image(mask.astype(np.int), np.eye(4))
    data_img = nibabel.Nifti1Image(np.reshape(y, (3, 3, 3, -1)), np.eye(4))
    return mask_img, data_img, X


def test_make_X_Y():
    stim_TR, TR = 0.1, 2
    stimulus = np.tile(np.arange(80)[:, None], (1, 1))
    fmri = np.tile(np.arange(0, 4)[:, None], (1, 1))
    x, y = prep.make_X_Y([stimulus, stimulus], [fmri, fmri], TR, stim_TR, lag_time=4, offset_stim=0, start_times=[0, 0])
    assert x.shape == (6, 40)
    assert y.shape == (6, 1)
    assert np.allclose(x[0], x[3])
    assert y[0] == y[3]
    # test removal of nans
    x, y = prep.make_X_Y([stimulus], [fmri], TR, stim_TR, lag_time=4, offset_stim=0, start_times=[0], remove_nans=False)
    assert x.shape == (4, 40)
    assert y.shape == (4, 1)
    assert np.isnan(x[0]).sum() == 20
    assert y[0] == 0
    # test stimulus offset 
    X = np.tile(np.arange(4000)[:, None], (1, 30))
    Y = np.tile(np.arange(0, 205)[:, None], (1, 10))
    x_lagged, y_lagged = prep.make_X_Y([X], [Y], TR, stim_TR, offset_stim=4., lag_time=6)
    assert x_lagged[0].max() == 59
    assert np.all(y_lagged[0] == 4)


def test_generate_lagged_stimulus():
    stim_TR = 0.1
    TR = 2
    X = np.tile(np.arange(4000)[:, None], (1, 30))
    Y = np.tile(np.arange(0, 205)[:, None], (1, 10))
    x_lagged = prep.generate_lagged_stimulus(X, Y.shape[0], TR, stim_TR, lag_time=6)
    assert x_lagged[2].max() == 59
    x_lagged = prep.generate_lagged_stimulus(X, Y.shape[0], TR, stim_TR, lag_time=4)
    assert x_lagged[1].max() == 39
    x_lagged = prep.generate_lagged_stimulus(X, Y.shape[0], TR, stim_TR, lag_time=2, start_time=2)
    assert x_lagged[1].max() == 19
    #test different TR, stim_TR combinations
    stim_TR = 0.2
    TR = 2
    X = np.tile(np.arange(2000)[:, None], (1, 30))
    Y = np.tile(np.arange(0, 200)[:, None], (1, 10))
    x_lagged = prep.generate_lagged_stimulus(X, Y.shape[0], TR, stim_TR, lag_time=6)
    assert x_lagged[2].max() == 29

def test_fmri_preprocessing():
    mask, data, _ = create_test_data()
    bold = prep.preprocess_bold_fmri(data)
    assert bold.shape == (100, 27)
    bold = prep.preprocess_bold_fmri(data, mask=mask)
    assert bold.shape == (100, 27)
    bold = prep.preprocess_bold_fmri(data, mask=mask, standardize='zscore')
    bold = prep.preprocess_bold_fmri(data, mask=mask, standardize='zscore', detrend=True)