from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    eps = np.random.normal(0,noise,n_samples)
    poly_func = lambda x:(x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    poly = np.vectorize(poly_func)
    uni_range = [-1.2,2]
    uni_data = np.random.uniform(low = uni_range[0],high=uni_range[1],size = n_samples)
    space = np.linspace(-1.2,2,1000)
    poly_vals = poly(uni_data)
    train_X,train_y,test_x,test_y = split_train_test(X = pd.DataFrame(uni_data),y = pd.DataFrame(poly_vals+eps))
    fig = go.Figure([go.Scatter(x=train_X[0],y = train_y[0],mode="markers",name ="train data"),
                     go.Scatter(x=test_x[0],y=test_y[0],mode= "markers",name = "test data"),
                     go.Scatter(x=space,y=poly(space),mode= "lines",name = "True function")])
    fig.update_layout()
    fig.show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    polynomial_degs = np.arange(11)
    avg_val_scores = []
    avg_train_scores = []
    for deg in polynomial_degs:
        avg_validation_score,avg_train_score = cross_validate(PolynomialFitting(deg),train_X,train_y,mean_square_error,cv=10)
        avg_val_scores.append(avg_validation_score)
        avg_train_scores.append(avg_train_score)

    fig2 = go.Figure([go.Scatter(x=polynomial_degs,y=avg_train_scores,name= "avg train scores",mode="lines+markers"),
                      go.Scatter(x=polynomial_degs,y=avg_val_scores,name = "avg validation scores",mode="lines+markers")])
    fig2.show()




    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = np.argmin(avg_val_scores)
    min_val_error = np.min(avg_val_scores)
    min_val_poly_est = PolynomialFitting(min_k)
    min_val_poly_est.fit(train_X, train_y)
    pred = min_val_poly_est.predict(test_x)
    print("min validation error: "+ str(min_val_error))
    print("min degree: "+ str(min_k))
    print("test error: "+ str(mean_square_error(test_y, pred)))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500,noise=10)
