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
    print ("("+str(uni_gauss.mu_) + ","+ str(uni_gauss.var_)+")")


    # Question 2 - Empirically showing sample mean is consistent
    # x stands for no. of samples

    x = np.arange(start = 10,stop = 1010,step = 10)
    y = np.absolute(np.array([uni_gauss.fit(norm_samples[:i]).mu_ for i in x])-10)

    uni_fig  = px.scatter(x = x,y= y)
    uni_fig.update_layout(title = "quality of estimator as function of quantity of samples")
    uni_fig.update_xaxes(title = "no. of samples")
    uni_fig.update_yaxes(title ="estimated-true distance")
    uni_fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(norm_samples)
    z = uni_gauss.pdf(sorted_samples)
    uni_fig.update_layout(title="pdf of samples")
    uni_fig.update_xaxes(title="no. of sample")
    uni_fig.update_yaxes(title="pdf value")
    uni_pdf_fig = px.scatter(x=sorted_samples,y= z)
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

    #  create all different combinations of mu from num_arr
    mu_array = np.zeros((40000,4))
    for i in range(200):
        for j in range(200):
            k = i*j
            mu_array[200*i+j][0] = num_arr[i]
            mu_array[200*i + j][2] = num_arr[j]

    x = np.unique(mu_array[:,0])
    y = np.unique(mu_array[:,2])
    samp_log_like = np.apply_along_axis(mult_gauss.log_likelihood,axis = 1,arr = mu_array,cov = cov,X =mult_norm_samples).reshape(200,200)

    ht_map = go.Figure(go.Heatmap(x=y, y=x, z=samp_log_like))
    ht_map.update_layout(title = "log-likelihood of multivariate Gaussian",xaxis_title  = "f3 value", yaxis_title = "f1 value")
    ht_map.show()
    # Question 6 - Maximum likelihood
    max_indices = np.unravel_index(samp_log_like.argmax(), samp_log_like.shape)
    max_likelihood_vals = (x[max_indices[0]], y[max_indices[1]])
    print("f1=" + str(max_likelihood_vals[0]) + " f3=" + str(max_likelihood_vals[1]))

# a test for log likelihood values of univariate gaussian
def test_univariate_likelihood():
    x= np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    uni_gauss = UnivariateGaussian()
    print(uni_gauss.log_likelihood(1,1,x))
    print(uni_gauss.log_likelihood(10, 1, x))
# a test for pdf values of multivariate gaussian
def test_multivariate_gaussian_pdf():
    mult_gauss = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    mult_norm_samples = np.random.multivariate_normal(mu, cov, 1000)
    mult_gauss.fit(mult_norm_samples)
    pdf_arr =mult_gauss.pdf(mult_norm_samples)
    print(pdf_arr)

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    #other tests I added:
    #test_multivariate_gaussian_pdf()
    #test_univariate_likelihood()

