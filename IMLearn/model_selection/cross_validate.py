from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    data = np.concatenate((X,y),axis=1)
    # np.random.shuffle(data)

    five_folds = np.array_split(data,cv)
    validation_scores = []
    train_scores = []
    for i in range(cv):
        validation = five_folds[i]
        if(cv-1>i > 0 ):
            remainder = np.concatenate(five_folds[:i]+five_folds[i+1:],axis =0)
        elif i==cv-1:
            remainder = np.concatenate(five_folds[:i],axis=0)
        else:
            remainder = np.concatenate(five_folds[i+1:],axis=0)
        estimator.fit(remainder[:,:-1],remainder[:,-1])
        train_pred = estimator.predict(remainder[:,:-1])
        train_score  = scoring(remainder[:,-1],train_pred)
        train_scores.append(train_score)
        val_pred = estimator.predict(validation[:,:-1])
        validation_score = scoring(validation[:,-1],val_pred)
        validation_scores.append(validation_score)
    avg_validation_score = np.mean(validation_scores)
    avg_train_score = np.mean(train_scores)

    return avg_train_score,avg_validation_score

# def get_validation_sets(data_copy,k):
#     val_train_sets = []
#     for i in range(k,1,-1):
#         mask = np.random.binomial(1,1/float(i),data.shape[0]).astype(bool)
#         validation_set,train_set = data_copy[mask],data_copy[~mask]
#         val_train_sets.append((train_set,validation_set))
#         data_copy = data_copy[~mask]
#     return val_train_sets

