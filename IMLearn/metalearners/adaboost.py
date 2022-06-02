import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics.loss_functions import misclassification_error as me


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models_ = []
        self.D_ = np.full(y.shape[0],1/y.shape[0])
        self.weights_ = []
        ones = np.ones(y.shape[0])
        for iteration in range(self.iterations_):
            # create estimator
            model = self.wl_()
            # fit estimator
            model._fit(X, self.D_ * y)
            # append to models list
            self.models_.append(model)
            # update weights
            model_prediction = model._predict(X)
            e_t  = np.dot((np.where(np.sign(model_prediction*y)<0,self.D_,0)),ones)
            w_t = 0.5*np.log((1/e_t)-1)
            self.weights_.append(w_t)
            sum_vec = np.exp(-w_t*np.sign(y)*model_prediction)
            s = self.D_ @ sum_vec
            opp = np.exp(-w_t*np.sign(y)*model_prediction)/s
            self.D_ = self.D_ * opp

        self.weights_ = np.array(self.weights_)

    def _predict(self, X):
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

        prediction = self.partial_predict(X,self.iterations_)
        return prediction

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
        return self.partial_loss(X,y,self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        partial_models = self.models_[:T+1]
        partial_weights = np.array(self.weights_)[:T+1,]
        partial_predictions =  np.array([model._predict(X) for model in partial_models])
        prediction = np.sign(partial_predictions.T @ partial_weights)
        return prediction

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        partial_prediction = self.partial_predict(X,T)
        error = me(y,partial_prediction)
        return error
