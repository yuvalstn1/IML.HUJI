from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso\
# TODO delete these imports
# from sklearn.model_selection import cross_validate as CV
# from sklearn.linear_model import Ridge

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
        avg_train_score,avg_validation_score = cross_validate(PolynomialFitting(deg),train_X,train_y,mean_square_error,cv=5)
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
    X,y = datasets.load_diabetes(return_X_y=True)
    fraction = n_samples/X.shape[0]
    train_x,train_y,test_x,test_y = split_train_test(pd.DataFrame(X),pd.Series(y),fraction)
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    hyper_param_ranges = [(0.000001,0.5),(1,10),(9000,10000)]
    for range in hyper_param_ranges:
        hyper_params = np.linspace(range[0],range[1],n_evaluations)
        l1_validation_scores = []
        l1_training_scores = []
        l2_validation_scores = []
        l2_training_scores = []
        best_l1_param = 0
        best_l1_score = None
        best_l2_param = 0
        best_l2_score = None

        for param in hyper_params:
            ridge_train_score, ridge_validation_score = cross_validate(RidgeRegression(lam=param), train_x.to_numpy(),
                                                                       train_y.to_numpy().reshape(n_samples,1),
                                                               mean_square_error, cv=5)
            lasso_train_score, lasso_validation_score = cross_validate(Lasso(alpha=param), train_x.to_numpy(),
                                                                       train_y.to_numpy().reshape(n_samples,1),
                                                                       mean_square_error, cv=5)
            l1_training_scores.append(lasso_train_score)
            l1_validation_scores.append(lasso_validation_score)
            l2_training_scores.append(ridge_train_score)
            l2_validation_scores.append(ridge_validation_score)
            if best_l1_score == None or best_l1_score>lasso_validation_score:
                best_l1_score = lasso_validation_score
                best_l1_param = param
            if best_l2_score == None or best_l2_score > ridge_validation_score:
                best_l2_score = ridge_validation_score
                best_l2_param = param



        fig = go.Figure([go.Scatter(x=hyper_params,y=l1_training_scores,name="lasso train err",mode="lines"),
                         go.Scatter(x=hyper_params,y=l1_validation_scores,name="lasso validation err",mode="lines"),
                         go.Scatter(x=hyper_params,y=l2_training_scores,name="ridge train err",mode="lines"),
                         go.Scatter(x=hyper_params,y=l2_validation_scores,name="ridge validation err",mode="lines")],
                    )
        fig.show()





    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso = Lasso(alpha=best_l1_param)
    best_ridge = RidgeRegression(lam=best_l2_param)
    ls = LinearRegression()
    best_ridge.fit(train_x,train_y)
    best_lasso.fit(train_x,train_y)
    ls.fit(train_x,train_y)
    print("least squares test err: ", ls.loss(test_x,test_y))
    print("ridge regression test err for reg param ",best_l2_param,": ", best_ridge.loss(test_x, test_y))
    print("lasso test err for reg param ",best_l1_param,": ",
          mean_square_error(test_y,best_lasso.predict(test_x)))

# def sk_cv(n_samples: int = 50, n_evaluations: int = 500):
#     X, y = datasets.load_diabetes(return_X_y=True)
#     fraction = n_samples / X.shape[0]
#     train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), fraction)
#     # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
#     hyper_param_ranges = [(0.00001, 1), (2, 3), (1, 10000)]
#     for range in hyper_param_ranges:
#         hyper_params = np.linspace(range[0], range[1], n_evaluations)
#         l1_validation_scores = []
#         l1_training_scores = []
#         l2_validation_scores = []
#         l2_training_scores = []
#         best_l1_param = 0
#         best_l1_score = None
#         best_l2_param = 0
#         best_l2_score = None
#
#         for param in hyper_params:
#             ridge_scores = CV(estimator=Ridge(alpha=param),X= train_x.to_numpy(),
#                                                                        y=train_y.to_numpy(),
#                               scoring='neg_mean_squared_error', cv=5, return_train_score=True)
#             ridge_validation_score, ridge_train_score = ridge_scores['test_score'], ridge_scores['train_score']
#             lasso_scores = CV(estimator=Lasso(alpha=param), X= train_x.to_numpy(),
#                               y=train_y.to_numpy(),
#                               scoring='neg_mean_squared_error', cv=5, return_train_score=True)
#             lasso_validation_score, lasso_train_score = lasso_scores['test_score'], lasso_scores['train_score']
#             l1_training_scores.append(lasso_train_score)
#             l1_validation_scores.append(lasso_validation_score)
#             l2_training_scores.append(ridge_train_score)
#             l2_validation_scores.append(ridge_validation_score)
#             # if best_l1_score == None or best_l1_score > lasso_validation_score:
#             #     best_l1_score = lasso_validation_score
#             #     best_l1_param = param
#
#         fig = go.Figure([go.Scatter(x=hyper_params, y=l1_training_scores, name="lasso train err", mode="lines"),
#                          go.Scatter(x=hyper_params, y=l1_validation_scores, name="lasso validation err", mode="lines"),
#                          go.Scatter(x=hyper_params, y=l2_training_scores, name="ridge train err", mode="lines"),
#                          go.Scatter(x=hyper_params, y=l2_validation_scores, name="ridge validation err", mode="lines")],
#                         )
#         fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500,noise=10)
    select_regularization_parameter()

