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


def test_lagging():
    import numpy as np
    stim_TR = 0.1
    TR = 2
    X = np.tile(np.arange(4000)[:, None], (1, 30))
    Y = np.tile(np.arange(0, 205)[:, None], (1, 10))
    x_lagged, y_lagged = prep.make_X_Y([X], [Y], TR, stim_TR, lag_time=6)
    assert x_lagged[0].max() == 59
    assert np.all(y_lagged[0] == 2)
    # test stimulus offset 
    x_lagged, y_lagged = prep.make_X_Y([X], [Y], TR, stim_TR, offset_stim=4., lag_time=6)
    assert x_lagged[0].max() == 59
    assert np.all(y_lagged[0] == 4)
    x_lagged, y_lagged = prep.make_X_Y([X], [Y], TR, stim_TR, lag_time=4)
    assert x_lagged[0].max() == 39
    assert np.all(y_lagged[0] == 1)
    x_lagged, y_lagged = prep.make_X_Y([X], [Y], TR, stim_TR, lag_time=2, start_times=[2])
    assert x_lagged[0].max() == 19
    assert np.all(y_lagged[0] == 1)


def test_fmri_preprocessing():
    mask, data, _ = create_test_data()
    bold = prep.preprocess_bold_fmri(data)
    assert bold.shape == (100, 27)
    bold = prep.preprocess_bold_fmri(data, mask=mask)
    assert bold.shape == (100, 27)
    bold = prep.preprocess_bold_fmri(data, mask=mask, standardize='zscore')
    bold = prep.preprocess_bold_fmri(data, mask=mask, standardize='zscore', detrend=True)