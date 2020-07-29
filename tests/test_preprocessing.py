#change to better import
from voxelwiseencoding import preprocessing as prep

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

