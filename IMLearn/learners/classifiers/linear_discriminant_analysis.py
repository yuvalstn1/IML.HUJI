from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error as me



class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.fitted_,self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = False,None, None, None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # fit the pi
        pi = []
        sum = []
        #number of samples
        m = y.shape[0]
        #get all different labels
        self.classes_ = np.unique(y)
        #find samples that accord to each label
        S = X.copy()
        for i in self.classes_:
            indices = np.nonzero(y == i)[0]
            nk = indices.shape[0]
            pi.append(nk)
            mu_k = np.mean(X[indices],axis = 0)
            sum.append(mu_k)
            S[indices] =S[indices] - mu_k
        self.pi_ = np.array(pi)/m
        self.mu_ = np.array(sum)
        self.cov_ = np.sum([np.outer(row,row) for row in S],axis =0)/m
        self._cov_inv = inv(self.cov_)
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
        if not self.fitted_:
            raise ValueError("Model must first be fitted before prediction")

        predictions = self.likelihood(X)
        prediction = np.argmax(predictions,axis = 1)
        return prediction


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = []
        for k,label in enumerate(self.classes_):
            k_likelihood = response_approx(self._cov_inv,self.mu_[k],X,self.pi_[k])
            likelihoods.append(k_likelihood)

        return np.array(likelihoods).T

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
        test_prediction = self._predict(X)
        return me(y,test_prediction)

def response_approx(inv_cov,mu_k,X,pi_k):
    a = inv_cov @ mu_k
    b = np.log(pi_k) - 0.5*mu_k @ inv_cov @ mu_k.T
    return  X @ a + b