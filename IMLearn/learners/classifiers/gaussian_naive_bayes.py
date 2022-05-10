from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error as me

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_,self._fitted = None, None, None, None,False

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = y.shape[0]
        sample_k_values = []
        self.classes_ = np.unique(y)
        for k in self.classes_:
            sample_k_values.append((y==k).astype(int))
        nk = np.array([np.count_nonzero(k_samples) for k_samples in sample_k_values ])
        self.pi_ = nk/m
        sample_k_ind = np.stack(sample_k_values)
        k_d_sum_matrice = sample_k_ind @ X
        for i,row in enumerate(k_d_sum_matrice): # note its k iterations we assume that the amount
            # of labels is much smaller than the amount of samples
            k_d_sum_matrice[i] = row/nk[i]
        self.mu_ = k_d_sum_matrice
        k_var = []
        # now we calculate the variance
        var_arr = []
        for k,row in enumerate(sample_k_values):
            t = np.square(X - self.mu_[k])
            k_samples = row @ t
            var_arr.append(k_samples/np.count_nonzero(row))
        k_d_var_matrice =  np.stack(var_arr)
        self.vars_= k_d_var_matrice
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
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        predictions = self.likelihood(X)
        prediction = np.argmax(predictions, axis=1)
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


        samples_likelihood = []
        for k,label in enumerate(self.classes_):
            k_likelihood = likelihood_approx(self.vars_[k],self.mu_[k],self.pi_[k],X)
            samples_likelihood.append(k_likelihood)

        return np.array(samples_likelihood).T

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
        return me(y, test_prediction)

def likelihood_approx(vars_k,mu_k,pi_k,X):
    k_likelihood = []
    for i,row in enumerate(X):
        sample_i = []
        for j,feature in enumerate(row):
            log_exp = np.square((X[i,j]-mu_k[j]))/2*vars_k[j]
            sample_i.append(np.log(pi_k)-0.5*np.log(vars_k[j])-0.5*np.log(2*np.pi)-log_exp)
        k_likelihood.append(np.sum(sample_i))
    return np.array(k_likelihood)

