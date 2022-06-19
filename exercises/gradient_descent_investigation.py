import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    recorded_values = []
    recorded_weights = []
    def generic_callback_func(gd: GradientDescent,weights : np.ndarray,values:np.ndarray,
                              grad:np.ndarray,iter: int,current_rate:float,delta:float):
        recorded_weights.append(weights)
        recorded_values.append(values)
        return
    return generic_callback_func,recorded_values,recorded_weights





def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [L1,L2]
    convergence_rates_plots = []
    min_loss = {}
    for objective in modules:
        if objective == L1:
            name = 'L1'
        else:
            name= 'L2'
        min = None
        for step_size in etas:
            callback,val_rec,weight_rec = get_gd_state_recorder_callback()
            GD = GradientDescent(learning_rate=FixedLR(step_size),callback=callback)
            GD.fit(f=objective(init),X=None,y=None)
            des_path_fig = plot_descent_path(objective,descent_path=np.array(weight_rec),title =f'descent path for module: {name},'
                                                                             f'fixed LR, step size: {step_size}')
            des_path_fig.show()

            conv_plot = go.Scatter(x=np.arange(1,GD.max_iter_+1),y= val_rec,mode= 'lines',
                                   name = f'{name},step size: {step_size}')
            convergence_rates_plots.append(conv_plot)
            cur_min = np.min(val_rec)
            if  min == None or cur_min<min:
                min = cur_min
        min_loss[name] = min
    conv_fig = go.Figure(data=convergence_rates_plots)
    conv_fig.show()
    print(min_loss)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    modules = [L1,L2]
    convergence_rates_plots = []
    min_loss = {}

    for objective in modules:
        min = None
        if objective == L1:
            name = 'L1'
        else:
            name= 'L2'
        for dec_rate in gammas:
            callback, val_rec, weight_rec = get_gd_state_recorder_callback()
            GD = GradientDescent(learning_rate=ExponentialLR(eta,dec_rate), callback=callback)
            GD.fit(f=objective(init), X=None, y=None)
            if dec_rate == 0.95:
                des_path_fig = plot_descent_path(objective, descent_path=np.array(weight_rec),
                                             title=f'descent path for module: {name},'
                                                   f'exponential LR, base rate: {eta},'
                                                   f'gamma: {dec_rate}')
                des_path_fig.show()
            if objective == L1:
                conv_plot = go.Scatter(x=np.arange(1, GD.max_iter_ + 1), y=val_rec, mode='lines',
                                   name=f'{name},base rate: {eta}'
                                        f'gamma: {dec_rate}')
                convergence_rates_plots.append(conv_plot)
            cur_min = np.min(val_rec)
            if min == None or cur_min < min:
                min = cur_min
        min_loss[name] = min
    conv_fig = go.Figure(data=convergence_rates_plots)
    conv_fig.show()
    print(min_loss)
    # Plot algorithm's convergence for the different values of gamma

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
