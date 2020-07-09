def test_lagging():
    import numpy as np
    stim_TR = 0.1
    TR = 2
    X = np.tile(np.arange(4000)[:, None], (1, 30))
    Y = np.tile(np.arange(0, 205)[:, None], (1, 10))
    x_lagged, y_lagged = prep.make_X_Y([X], [Y], TR, stim_TR)
    assert x_lagged[0].max() == 39
    assert x_lagged[-1]
