# AUTOGENERATED! DO NOT EDIT! File to edit: encoding.ipynb (unless otherwise specified).

__all__ = ['product_moment_corr', 'get_model_plus_scores', 'BlockMultiOutput']

# Cell
#export
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import warnings
import copy
from joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputRegressor, _fit_estimator
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import RegressorMixin

def product_moment_corr(x,y):
    '''Product-moment correlation for two ndarrays x, y'''
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    n = x.shape[0]
    r = (1/(n-1))*(x*y).sum(axis=0)
    return r

# Cell

def get_model_plus_scores(X, y, estimator=None, cv=None, scorer=None,
                          voxel_selection=True, validate=True, **kwargs):
    '''Returns multiple estimator trained in a cross-validation on n_splits of the data and scores on the left-out folds

    Parameters

        X : ndarray of shape (samples, features)
        y : ndarray of shape (samples, targets)
        estimator : None or estimator object that implements fit and predict
                    if None, uses RidgeCV per default
        cv : int, None, or a cross-validation object that implements a split method, default is None, optional.
             int specifies the number of cross-validation splits of a KFold cross validation
             None defaults to a scikit-learn KFold cross-validation with default settings
             a scikit-learn-like cross-validation object needs to implement a split method for X and y
        scorer : None or any sci-kit learn compatible scoring function, optional
                 default uses product moment correlation
        voxel_selection : bool, optional, default True
                          Whether to only use voxels with variance larger than zero.
                          This will set scores for these voxels to zero.
        validate : bool, optional, default True
                     Whether to validate the model via cross-validation
                     or to just train the estimator
                     if False, scores will be computed on the training set
        kwargs : additional parameters that will be used to initialize RidgeCV if estimator is None
    Returns
        tuple of n_splits estimators trained on training folds or single estimator if validation is False
        and scores for all concatenated out-of-fold predictions'''
    from sklearn.utils.estimator_checks import check_regressor_multioutput
    if scorer is None:
        scorer = product_moment_corr
    if cv is None:
        cv = KFold()
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
    models = []
    score_list = []
    if estimator is None:
        estimator = RidgeCV(**kwargs)

    if voxel_selection:
        voxel_var = np.var(y, axis=0)
        y = y[:, voxel_var > 0.]
    if validate:
        for train, test in cv.split(X, y):
            models.append(copy.deepcopy(estimator).fit(X[train], y[train]))
            if voxel_selection:
                scores = np.zeros_like(voxel_var)
                scores[voxel_var > 0.] =  scorer(y[test], models[-1].predict(X[test]))
            else:
                scores = scorer(y[test], models[-1].predict(X[test]))
            score_list.append(scores[:, None])
        score_list = np.concatenate(score_list, axis=-1)
    else:
        models = estimator.fit(X, y)
        score_list = scorer(y, estimator.predict(X))
    return models, score_list

# Cell

class BlockMultiOutput(MultiOutputRegressor, RegressorMixin):
    """Multi target regression with block-wise fit
    This strategy consists of splitting the targets in blocks and fitting one regressor per block.
    The estimator used needs to natively support multioutput.

    Parameters

        estimator : estimator object
            An estimator object implementing `fit` and `predict` and supporting multioutput.
        n_blocks : int, optional, default=10
            The number of blocks for the target variable.
            This is a split along *targets* (columns of the array), not observations (rows of the array).
        n_jobs : int, optional, default=1
            The number of jobs to run in parallel for `fit`. If -1,
            then the number of jobs is set to the number of cores.
            When individual estimators are fast to train or predict
            using `n_jobs>1` can result in slower performance due
            to the overhead of spawning processes.
    """

    def __init__(self, estimator, n_blocks=10, n_jobs=1):
        self.estimator = estimator
        self.n_blocks = n_blocks
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """ Fit the model to data.
        Fit a separate model for each chunk of output.

        Parameters

            X : (sparse) array-like, shape (n_samples, n_features)
                Data.
            y : (sparse) array-like, shape (n_samples, n_outputs)
                Multi-output targets. An indicator matrix turns on multilabel
                estimation.
            sample_weight : array-like, shape = (n_samples) or None
                Sample weights. If None, then samples are equally weighted.
                Only supported if the underlying regressor supports sample
                weights.

        Returns

            self : object
                Returns self
        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")
        kfold = KFold(n_splits=self.n_blocks)
        smpl_X, smpl_y = np.zeros((y.shape[1],1)), np.zeros((y.shape[1],1))
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, block], sample_weight)
            for _, block in kfold.split(smpl_X, smpl_y))
        return self

    def partial_predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable block and yields predictions for each block as an iterator.

        Parameters

        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        Returns

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        for estimator in self.estimators_:
            yield estimator.predict(X)

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable block.

        Parameters

            X : (sparse) array-like, shape (n_samples, n_features)
                Data.

        Returns

            y : (sparse) array-like, shape (n_samples, n_outputs)
                Multi-output targets predicted across multiple predictors.
                Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X)
            for e in self.estimators_)

        return np.hstack(y)

    def score(self, X, y):
        """Returns the correlation of the prediction with the target for each output.

        Parameters

            X : array-like, shape (n_samples, n_features)
                Test samples.
            y : array-like, shape (n_samples) or (n_samples, n_outputs)
                True values for X.

        Returns

            score : float
                Correlation of self.predict(X) wrt. y.
        """
        from sklearn.preprocessing import StandardScaler
        from itertools import izip
        kfold = KFold(n_splits=self.n_blocks)
        smpl_X, smpl_y = np.zeros((y.shape[1],1)), np.zeros((y.shape[1],1))
        scores = []
        for prediction, (_, block) in izip(self.partial_predict(X), kfold.split(smpl_X, smpl_y)):
            mx = StandardScaler().fit_transform(prediction).astype('float32')
            my = StandardScaler().fit_transform(y[:, block]).astype('float32')
            n = mx.shape[0]
            r = (1/(n-1))*(mx*my).sum(axis=0)
            scores.append(r)
        return np.concatenate(scores)
