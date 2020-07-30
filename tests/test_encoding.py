from voxelwiseencoding import encoding as enc
import numpy as np

def create_encoding_test_data():
    '''Creates toy stimulus and fmri data to test voxelwise encoding models.'''
    X = np.reshape(np.random.randn(1000, 5), (100, -1))
    betas = np.random.randn(X.shape[-1], 27) * 2
    # create different noise levels so that we find different regularization parameters
    y = X.dot(betas) + np.random.rand(100, 27).dot(np.diag([0.1]*26 + [10]))
    return X, y


def test_encoding():
    X, y  = create_encoding_test_data()
    ridges, scores = enc.get_ridge_plus_scores(X, y, n_splits=2)
    assert len(ridges) == 2
    assert scores.shape == (27,)
    print(ridges[0].alpha)
    print(ridges[1].alpha)