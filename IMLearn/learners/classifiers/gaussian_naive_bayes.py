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
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

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
        for i,row in enumerate(k_d_sum_matrice):
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
        class_results = []
        for i in self.classes_:
            class_results.append(self.pi_[i]*multi_var_gauss_pdf(self.vars_,self.mu_,X))
        predictions = np.argmax(class_results)
        return predictions


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

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        sample_likelihood = []
        for sample in X:
            sample_class_likelihood = []
            for k in self.classes_:
                sample_class_likelihood.append(self.pi_[k] * multi_var_gauss_pdf(cov=self.cov_, mu=self.mu_, X=sample))
            sample_likelihood.append(sample_class_likelihood)
        return np.ndarray(sample_likelihood)

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

def multi_var_gauss_pdf(cov,mu,X):
    constants_operand = (1 / np.sqrt(np.power(2 * np.pi, X.shape[1]) * det(cov)))
    x_mu = X - mu
    exp_operand = np.exp((-0.5) * (x_mu @ inv(cov) @ x_mu.T))
    pdf_arr = constants_operand * exp_operand
    pdf_arr = np.diagonal(pdf_arr)
    return pdf_arr