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
        S = X
        for i in self.classes_:
            #TODO fix indices
            indices = np.nonzero(y == i)[0]
            nk = indices.shape[0]
            pi.append(nk)
            mu_k = np.sum(X[indices,:])/nk
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
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        class_results = []
        for i in self.classes_:
            class_results.append(self.pi_[i]*multi_var_gauss_pdf(self.cov_,self.mu_,X))
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

        sample_likelihood = []
        for sample in X:
            sample_class_likelihood = []
            for k in self.classes_:
                sample_class_likelihood.append(self.pi_[k]*multi_var_gauss_pdf(cov = self.cov_,mu = self.mu_,X = sample))
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
        return me(y,test_prediction)

def multi_var_gauss_pdf(cov,mu,X):
    constants_operand = (1 / np.sqrt(np.power(2 * np.pi, X.shape[1]) * det(cov)))
    x_mu = X - mu
    exp_operand = np.exp((-0.5) * (x_mu @ inv(cov) @ x_mu.T))
    pdf_arr = constants_operand * exp_operand
    pdf_arr = np.diagonal(pdf_arr)
    return pdf_arr