from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    csv_data = pd.read_csv(filename)
    csv_data = csv_data.dropna(axis = 0,subset = ['price'])
    response = csv_data['price']

    csv_data.drop(['price','id','date'],axis = 1,inplace= True)
    csv_data = pd.get_dummies(csv_data,prefix='zipcode num ',columns=['zipcode'])
    #csv_data['date']= np.float_(csv_data['date'].str.replace('T000000',''))/10000

    #get mean values of every column fill blank spaces with mean value of column
    mean_values = {column: csv_data[column].mean() for column in csv_data.columns
                   if column not in {'id','date','zipcode','yr_renovated'}}
    csv_data.fillna(value = mean_values)
    csv_data.fillna(0)


    return csv_data,response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    col_corr_array = np.array([(column,(X[column].cov(y)/((X[column]).std()*y.std()))) for column in X.columns])

    for column,corr in col_corr_array:
        fig  = px.scatter(x = X[column],y =y,title = column+"-price pearson correlation: "+corr)

        pio.write_image(fig,file = output_path+str(column)+"fig.png")




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df,response = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    #feature_evaluation(df,response,"C:/Users/yuval/Desktop/IML.HUJI/junk_folder/")

    # Question 3 - Split samples into training- and testing sets.
    train_samples,train_response,test_samples,test_response = split_train_test(df,response,0.75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    frac_array =[]
    mean_loss_arr = []
    var_loss_arr = []
    lin_reg = LinearRegression()
    for i in range(10,101):
        frac = i
        frac_array.append(frac)
        loss_p = []
        for x in range(10):
            train_x,train_y,test_x,test_y = split_train_test(train_samples,train_response,float(frac)/100)
            lin_reg._fit(train_x,train_y)
            loss = lin_reg._loss(test_samples,test_response)
            loss_p.append(loss)
        mean_loss = np.mean(loss_p)
        var_loss = np.std(loss_p)
        var_loss_arr.append(var_loss)
        mean_loss_arr.append(mean_loss)
    var_loss_arr = np.array(var_loss_arr)
    mean_loss_arr = np.array(mean_loss_arr)
    fig = go.Figure(data= [go.Scatter(x = frac_array,y=mean_loss_arr,name = "mean_loss",mode="markers+lines"),
                           go.Scatter(x = frac_array,y=mean_loss_arr + 2*var_loss_arr, mode="lines", fill='tonexty',line=dict(color="lightgrey"), showlegend=False),
                            go.Scatter(x=frac_array, y=mean_loss_arr - 2*var_loss_arr, mode="lines", fill='tonexty',line=dict(color="lightgrey"), showlegend=False),
                           ],
                    layout = go.Layout(title_text="MSE as function of p% of training data",
                         xaxis={"title": "p% of training data"},
                         yaxis={"title": "mean_loss on test set"})
                    )

    fig.show()

