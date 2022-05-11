import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    #todo fix nlearners to 250

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_ensemble = AdaBoost(wl = DecisionStump,iterations=n_learners)
    adaboost_ensemble._fit(train_X,train_y)
    train_errors = np.array([adaboost_ensemble.partial_loss(train_X,train_y,t) for t in range(0,n_learners)])
    test_errors = np.array([adaboost_ensemble.partial_loss(test_X,test_y,t) for t in range(0,n_learners)])
    iterations = np.arange(n_learners)

    adaboost_fig = go.Figure(data=[go.Scatter(x=iterations,y=train_errors,name = "train error"),
                                   go.Scatter(x=iterations,y=test_errors,name="test error")])
    adaboost_fig.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    # T = [1,5,10,20]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    const_partial_predict  = lambda t:lambda train: adaboost_ensemble.partial_predict(train,t)
    const_partial_loss = lambda t:lambda train,test: adaboost_ensemble.partial_loss(train,test,t)
    y_preds= [const_partial_predict(t) for t in T]

    boundaries_fig = make_subplots(rows = 2,cols = 2)
    for i,prediction in enumerate(y_preds):
            boundaries_fig.add_traces([decision_surface(y_preds[i],lims[0],lims[1]),
                                      go.Scatter(x= test_X[:,0],y=test_X[:,1],mode="markers",marker=dict(color=test_y))],
                                     rows = (i//2)+1,cols= (i % 2)+1)

    boundaries_fig.show()
    # Question 3: Decision surface of best performing ensemble
    y_losses = [const_partial_loss(t) for t in T]
    min_t = T[np.argmin([loss(test_X,test_y) for loss in y_losses])]
    best_acc = accuracy(test_y,adaboost_ensemble.partial_predict(test_X,min_t))
    best_boundary = go.Figure(data = [decision_surface(const_partial_predict(min_t),lims[0],lims[1]),
                                      go.Scatter(x= test_X[:,0],y=test_X[:,1],mode="markers",marker=dict(color=test_y))]
                                     )
    best_boundary.update_layout(title = "best boundary in iteration: "+str(min_t)+ " accuracy: " + str(best_acc))
    best_boundary.show()
    # Question 4: Decision surface with weighted samples
    last_weights = adaboost_ensemble.D_
    last_weights = last_weights/np.max(last_weights) * 5
    size_plot = go.Figure(data = [decision_surface(const_partial_predict(adaboost_ensemble.iterations_),lims[0],lims[1]),
                                      go.Scatter(x= train_X[:,0],y=train_X[:,1],mode="markers",
                                                 marker=dict(color=train_y,size = last_weights))]
                                     )
    size_plot.show()



if __name__ == '__main__':
    np.random.seed(0)
    # iterations = 20
    # fit_and_evaluate_adaboost(0,iterations)
    # fit_and_evaluate_adaboost(0.4,iterations)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)