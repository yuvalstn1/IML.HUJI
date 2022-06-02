from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error as me


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        sign = 1
        min = -1
        for feature in range(X.shape[1]):
            pos_feature_thresh,pos_feature_min_error = self._find_threshold(X[:,feature],y,sign)
            neg_feature_thresh,neg_feature_min_error = self._find_threshold(X[:,feature],y,-sign)
            if pos_feature_min_error > neg_feature_min_error:
                minsign = -1
                feature_min_error = neg_feature_min_error
                feature_thresh = neg_feature_thresh
            else:
                minsign = 1
                feature_min_error = pos_feature_min_error
                feature_thresh = pos_feature_thresh
            if min ==-1:
                min_feature = feature
                min_threash = feature_thresh
                min = feature_min_error
                self.sign_ = minsign
            if feature_min_error< min:
                min_feature = feature
                min_threash = feature_thresh
                min = feature_min_error
                self.sign_ = minsign
        self.j_ = min_feature
        self.threshold_ =  min_threash
        self.fitted_ = True






    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        prediction = np.where(X[:,self.j_] < self.threshold_,-self.sign_,self.sign_)
        return prediction

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        min = -1
        best_thresh = 0
        ones = np.ones(values.shape[0])
        unique_values = np.unique(values)
        for value in unique_values:
            threshvslabels_vals = np.sign(labels*np.where(values < value,-sign,sign))
            error = np.dot(np.where(threshvslabels_vals<0,np.abs(labels),0),ones)
            if min == -1:
                min = error
                best_thresh = value
            if error < min:
                min = error
                best_thresh = value
        return best_thresh,min


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return me(y,self._predict(X))
