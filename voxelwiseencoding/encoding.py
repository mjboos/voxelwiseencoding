# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import sys
from os.path import join
import glob
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import joblib
from skimage.util.shape import view_as_windows
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

__all__ = ['get_ridge_plus_scores', 'ridge_gridsearch_per_target']

def get_ridge_plus_scores(X, y, alphas=None, n_splits=8, scorer=None, **kwargs):
    '''Returns ridge regressions trained in a cross-validation on n_splits of the data and scores on the left-out folds

    Parameters
    ----------
    X : ndarray of shape (samples, features)
    y : ndarray of shape (samples, targets)
    alphas : None or list of floats, optional
             Regularization parameters to be used for Ridge regression
    n_splits : int, optional
    scorer : None or any sci-kit learn compatible scoring function, optional
             default uses r2_score
    kwargs : additional arguments transferred to ridge_gridsearch_per_target

    Returns
    -------
    tuple of n_splits RidgeCV estimators trained on training folds
    and scores for all concatenated out-of-fold predictions'''
    if scorer is None:
        scorer = lambda x, y : r2_score(x, y, multioutput='raw_values')
    kfold = KFold(n_splits=n_splits)
    if alphas is None:
        alphas = [1000]
    ridges = []
    predictions = []
    ridges = []
    for train, test in kfold.split(X, y):
        ridges.append(ridge_gridsearch_per_target(X[train], y[train], alphas, **kwargs))
        predictions.append(ridges[-1].predict(X[test]))
    return ridges, scorer(y, np.vstack(predictions))

def ridge_gridsearch_per_target(X, y, alphas, n_splits=7, **kwargs):
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
    from sklearn.model_selection import KFold
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

