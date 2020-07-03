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

def get_ridge_plus_scores(X, y, alphas=None, n_splits=8, scorer=None, **kwargs):
    '''8-fold CV with inner CV gridsearch over alphas for Ridge(**kwargs) of X, y.
    Returns concatenated test-set predictions and 8 RidgeCV objects trained on the training folds'''
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
    '''Does gridsearch for alphas on X, y with cv and returns refit Ridge with best alphas'''
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

