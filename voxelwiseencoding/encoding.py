# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

__all__ = ['get_ridge_plus_scores', 'ridge_gridsearch_per_target']

def product_moment_corr(x,y):
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    n = x.shape[0]
    r = (1/(n-1))*(x*y).sum(axis=0)
    return r

def get_ridge_plus_scores(X, y, alphas=None, n_splits=8, scorer=None, voxel_selection=True, **kwargs):
    '''Returns ridge regressions trained in a cross-validation on n_splits of the data and scores on the left-out folds

    Parameters
    ----------
    X : ndarray of shape (samples, features)
    y : ndarray of shape (samples, targets)
    alphas : None or list of floats, optional
             Regularization parameters to be used for Ridge regression
    n_splits : int, optional
    scorer : None or any sci-kit learn compatible scoring function, optional
             default uses product moment correlation
    voxel_selection : bool, optional, default True
                      Whether to only use voxels with variance larger than zero.
                      This will set scores for these voxels to zero.
    kwargs : additional arguments transferred to ridge_gridsearch_per_target

    Returns
    -------
    tuple of n_splits Ridge estimators trained on training folds
    and scores for all concatenated out-of-fold predictions'''
    if scorer is None:
        scorer = product_moment_corr
    kfold = KFold(n_splits=n_splits)
    if alphas is None:
        alphas = [1000]
    ridges = []
    score_list = []
    # TODO: likely memory inefficient, should be changed
    if voxel_selection:
        voxel_var = np.var(y, axis=0)
        y = y[:, voxel_var > 0.]
    for train, test in kfold.split(X, y):
        ridges.append(ridge_gridsearch_per_target(X[train], y[train], alphas, **kwargs))
        if voxel_selection:
            scores = np.zeros_like(voxel_var)
            scores[voxel_var > 0.] =  scorer(y[test], ridges[-1].predict(X[test]))
        else:
            scores = scorer(y[test], ridges[-1].predict(X[test]))
        score_list.append(scores[:, None])
    return ridges, np.concatenate(score_list, axis=-1)

def ridge_gridsearch_per_target(X, y, alphas, n_splits=5, **kwargs):
    '''Runs Ridge gridsearch across alphas for each target in y

    Parameters
    ----------
    X : ndarray of shape (samples, features)
    y : ndarray of shape (samples, targets)
    alphas : None or list of floats, optional
             Regularization parameters to be used for Ridge regression
    n_splits : int, optional
    kwargs : keyword parameters to be transferred to Ridge regression

    Returns
    -------
    Ridge regression trained on X, y with optimal alpha per target
    determined by KFold cross-validation
    '''
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    cv_results = {'alphas': []}
    cv = KFold(n_splits=n_splits)
    for alpha in alphas:
        scores = []
        for train, test in cv.split(X, y):
            ridge = Ridge(alpha=alpha, **kwargs)
            scores.append(mean_squared_error(y[test], ridge.fit(X[train], y[train]).predict(X[test]),
                              multioutput='raw_values'))
        scores = np.vstack(scores).mean(axis=0)
        cv_results['alphas'].append(scores)
    cv_results['alphas'] = np.vstack(cv_results['alphas'])
    best_alphas = np.array(alphas)[np.argmin(cv_results['alphas'], axis=0)]
    return Ridge(alpha=best_alphas, **kwargs).fit(X, y)

