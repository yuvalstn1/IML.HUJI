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
    mu = np.array([0,0,4,0])
    cov = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    mult_norm_samples = np.random.multivariate_normal(mu,cov,1000)
    mult_gauss = MultivariateGaussian()
    mult_gauss.fit(mult_norm_samples)
    print(mult_gauss.mu_)
    print(mult_gauss.cov_)

    # Question 5 - Likelihood evaluation
    num_arr = np.linspace(-10,10,200)

    mu_array = np.zeros((400000,4))
    for i in range(200):
        for j in range(200):
            k = i*j
            mu_array[200*i+j][0] = num_arr[i]
            mu_array[200*i + j][2] = num_arr[j]
    likelihood_func = np.vectorize(mult_gauss.log_likelihood)
    sample_log_likelikhood = likelihood_func(mu_array,cov,mult_norm_samples)
    print("hello")
    # Question 6 - Maximum likelihood
    #raise NotImplementedError()

def mat_mean():
    mat= np.array ([[10,7,8],[3,0,3],[1,0,1]])
    print(np.mean(mat, axis = 1))

if __name__ == '__main__':
    np.random.seed(0)
    #test_univariate_gaussian()
    test_multivariate_gaussian()
    #mat_mean()
