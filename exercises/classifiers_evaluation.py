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
        line_fig.show()


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
                                                                                                                "gaussian, acc = " + str(gnb_accuracy)])
        fig.add_trace(go.Scatter(x = samples[:,0] ,y=samples[:,1],mode="markers",marker=dict(color=y_pred_lda, symbol=response)),
                      row = 1,col=1)
        fig.add_trace(go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers",marker=dict(color=y_pred_gnb, symbol=response)),
                      row = 1,col=2)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
