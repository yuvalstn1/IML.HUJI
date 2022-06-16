from __future__ import annotations
import numpy as np
from IMLearn.metrics.loss_functions import *
from IMLearn.model_selection.cross_validate import *
import IMLearn.learners.regressors.polynomial_fitting as pf

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso\
# TODO delete these imports
from sklearn.model_selection import cross_val_score as CV
# from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as ls

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mean_and_var(x: np.ndarray):
    y = np.square(x-2)
    return y

def mat_mean():
    mat= np.array ([[1,2,3][3,3,3][1,1,1]])
    print(np.mean(mat, axis = 1))



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
        reg_est = ls()
        avg_validation_score = np.mean(CV(reg_est,PolynomialFeatures(deg).fit_transform(train_X),train_y,cv=5))
        avg_val_scores.append(avg_validation_score)

    # fig2 = go.Figure([go.Scatter(x=polynomial_degs,y=avg_train_scores,name= "avg train scores",mode="lines+markers"),
    #                   go.Scatter(x=polynomial_degs,y=avg_val_scores,name = "avg validation scores",mode="lines+markers")])
    # fig2.update_xaxes(title = "polynomial degree")
    # fig2.update_yaxes(title = "MSE")
    # fig2.show()




    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = np.argmin(avg_val_scores)
    min_val_error = np.min(avg_val_scores)
    reg_est = ls()
    min_val_poly_est = reg_est.fit(PolynomialFeatures(min_k).fit_transform(train_X),train_y)
    pred = reg_est.predict(PolynomialFeatures(min_k).fit_transform(test_x))
    print("min validation error: "+ str(min_val_error))
    print("min degree: "+ str(min_k))
    print("test error: "+ str(mean_square_error(test_y, pred)))




if __name__ == "__main__":
    # x = np.array([4,1,4])
    # print(mean_and_var(x))
    # y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    # y_pred = np.array(
    #     [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    # print(mean_square_error(y_true,y_pred))
    # pass
    # y1 = np.array([[1,0,1],[1,1,1]])
    # y2 = np.array([[1,0],[1,1]])
    # ls = [y1,y2]
    # print((y1 == y2).astype(int))
    # print(np.stack(ls))
    # print(y2 * y1)
    # y1 = np.array([2,-1,-4,3,4])
    # y2 = np.array([-1,1,-1,1,2])
    # costs = np.array([1,2,3])
    # k =3
    # # cross_validate(pf.PolynomialFitting(k),y1,y2,mean_square_error)
    # y1 = [y1]
    # y2 = [y2]
    # print(y1+y2)
    #
    # print(np.argmin(y1))
    select_polynomial_degree(noise=0)