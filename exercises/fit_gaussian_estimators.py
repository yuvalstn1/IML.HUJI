from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    norm_samples = np.random.normal(10,1,1000)
    #initialize univariate gaussian
    uni_gauss  = UnivariateGaussian()
    #fit our gaussian to our samples
    uni_gauss.fit(norm_samples)



    # Question 2 - Empirically showing sample mean is consistent
    # x stands for no. of samples

    x = np.arange(start = 10,stop = 1010,step = 10)
    #y_temp = np.array([norm_samples[:i] for i in x])
    y = np.absolute(np.array([uni_gauss.fit(norm_samples[:i]).mu_ for i in x])-10)
    #print(y_temp)

    uni_fig  = px.scatter(x = x,y= y)
    uni_fig.update_layout(title = "quality of estimator as function of quantity of samples")
    uni_fig.update_xaxes(title = "no. of samples")
    uni_fig.update_yaxes(title ="estimated-true distance")
    uni_fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    z = uni_gauss.pdf(np.sort(norm_samples))
    uni_fig.update_layout(title="pdf of samples")
    uni_fig.update_xaxes(title="no. of sample")
    uni_fig.update_yaxes(title="pdf value")
    uni_pdf_fig = px.scatter(y= z)
    uni_pdf_fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    #test_multivariate_gaussian()
