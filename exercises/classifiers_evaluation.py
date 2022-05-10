from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from IMLearn.metrics.loss_functions import accuracy
from math import atan2

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    full_data = np.load(filename)
    y = full_data[:,2]
    x  = full_data[:,:2]
    return x,y



def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X,y =load_dataset("../datasets/"+f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def los(perc: Perceptron,samples,classification):
            loss = perc._loss(samples,classification)
            losses.append(loss)
        yello = Perceptron(callback=los,include_intercept=True)._fit(X,y)


        # Plot figure
        iter_num = np.arange(len(losses))

        line_fig = px.line(x=iter_num,y = losses)
        line_fig.update_xaxes(title = "iterations")
        line_fig.update_yaxes(title="loss")
        line_fig.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray,name: str):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",marker_color="black",name =name)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        samples,response =load_dataset("../datasets/"+f)

        # Fit models and predict over training set
        lda_model,gnb_model = LDA(),GaussianNaiveBayes()
        lda_model._fit(samples,response)
        gnb_model._fit(samples,response)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        y_pred_lda = lda_model.predict(samples)
        y_pred_gnb = gnb_model.predict(samples)
        lda_accuracy = accuracy(response,y_pred_lda)
        gnb_accuracy = accuracy(response,y_pred_gnb)

        fig = make_subplots(rows = 1, cols = 2, horizontal_spacing = 0.01, vertical_spacing=.03,subplot_titles=["lda, acc = " + str(lda_accuracy) ,
                                                                                                                "gaussian, acc = " + str(gnb_accuracy)],x_title = "feature1",
                            y_title="feature2")
        fig.add_trace(go.Scatter(x = samples[:,0] ,y=samples[:,1],mode="markers",marker=dict(color=y_pred_lda, symbol=response),name = "lda"),
                      row = 1,col=1)
        fig.add_traces([get_ellipse(lda_model.mu_[0],lda_model.cov_,"class0"),get_ellipse(lda_model.mu_[1],lda_model.cov_,"class1"),get_ellipse(lda_model.mu_[2],lda_model.cov_,"class2")],rows= 1,cols=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_[:,0],y=lda_model.mu_[:,1],mode="markers",marker=dict(color='black',symbol ='x')),row=1,col=1)
        fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers",marker=dict(color=y_pred_gnb, symbol=response),name = "gnb"),
                      row = 1,col=2)
        fig.add_traces([get_ellipse(gnb_model.mu_[0], np.diag(gnb_model.vars_[0]), "class0"),
                        get_ellipse(gnb_model.mu_[1], np.diag(gnb_model.vars_[1]), "class1"),
                        get_ellipse(gnb_model.mu_[2], np.diag(gnb_model.vars_[2]), "class2")], rows=1, cols=2)
        fig.add_trace(go.Scatter(x=gnb_model.mu_[:, 0], y=gnb_model.mu_[:, 1], mode="markers",
                                 marker=dict(color='black', symbol='x')), row=1, col=2)
        fig.update_layout(title = f)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
