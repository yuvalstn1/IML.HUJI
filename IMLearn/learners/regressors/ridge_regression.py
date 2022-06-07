from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """

        data = X
        lam_i = self.lam_*np.identity(X.shape[1])
        if self.lam_ == 0:
           # we have a regular linear regression problem
           if self.include_intercept_:
               added_intercept = np.ones(X.shape[0]).T.reshape(X.shape[0], 1)
               data = np.concatenate((added_intercept, data), axis=1)
           self.coefs_ = np.linalg.pinv(data) @ y
        else:
            d = X.shape[1]
            ols_x = np.concatenate([X,np.sqrt(lam_i)],axis=0)
            ols_y = np.concatenate([y,np.zeros(d)])
            if self.include_intercept_:
                added_intercept = np.ones((ols_x.shape[0],1))
                ols_x = np.concatenate((added_intercept, ols_x), axis=1)
            self.coefs_ = np.linalg.pinv(ols_x) @ ols_y


        self.fitted_ = True



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        data = X
        if self.include_intercept_ :
            added_intercept = np.ones(X.shape[0]).T.reshape(X.shape[0], 1)
            data = np.concatenate((added_intercept, X), axis=1)
        return data @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return np.mean((y-self._predict(X))**2)

def centralize_data(X: np.ndarray,y: np.ndarray):
    features = X.shape[1]
    x_c = X
    std_deviations = []
    for i in range(features):
        std = np.std(x_c[i])
        x_c[i] = (x_c[i]- np.mean(x_c[i]))/std
        std_deviations.append(std)
    y_c = (y - np.mean(y))
    return x_c,y_c,np.array(std_deviations)