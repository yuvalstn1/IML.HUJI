import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

#average_tempratures in each city in our data in case we have bad temp values
# harvested from https://en.climate-data.org/
average_temp_capetown = {1:20,2:20,3:19,4:17,5:15,6:14,7:13,8:13,9:14,10:16,11:17,12:19}
average_temp_amsterdam = {1:4,2:4,3:6,4:10,5:13,6:16,7:18,8:18,9:15,10:12,11:8,12:5}
average_temp_tel_aviv = {1:13,2:14,3:16,4:19,5:22,6:25,7:27,8:27,9:26,10:23,11:19,12:15}
average_temp_amman = {1:7,2:9,3:12,4:16,5:21,6:24,7:26,8:26,9:23,10:20,11:14,12:9}
city_dicts = {"Capetown":average_temp_capetown,
           "Amsterdam":average_temp_amsterdam,
           "Tel Aviv":average_temp_tel_aviv,
           "Amman":average_temp_amman}




def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    temp_data = pd.read_csv(filename,parse_dates=['Date'])
    # get day of year

    temp_data['DayOfYear'] = temp_data['Date'].dt.dayofyear
    temp_data.drop(['Date'],axis=1,inplace=True)
    #fix bad temprature by putting average temprature according to samples month threshold is 30 degrees
    # more or less than average temprature of month
    temp_data['Temp'] = temp_data.apply(fix_temp_value,axis = 1)

    #uncomment if need to use categorical data like country or city names

    # temp_data = pd.get_dummies(temp_data, prefix='country_name', columns=['Country'])
    # temp_data = pd.get_dummies(temp_data, prefix='city_name ', columns=['City'])


    samples = temp_data.drop('Temp', axis = 1)
    resp = temp_data['Temp']
    return samples,resp

def fix_temp_value(row):
    city = row['City']
    city_dict = city_dicts[city]
    month = int(row['Month'])
    temprature = float(row['Temp'])
    threshold = 30
    if temprature > city_dict[month] + threshold or temprature < city_dict[month] - threshold:
        return city_dict[month]
    else:
        return temprature


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp_samples,response = load_data('../datasets/City_Temperature.csv')
    full_data = pd.concat((temp_samples,response),axis =1)
    full_data['Temp'] = full_data.apply(fix_temp_value,axis = 1)

    # Question 2 - Exploring data for specific country
    israel_data = full_data[full_data['Country'] == "Israel"]
    #make sure coloring is by year and discrete
    israel_data["Year"] = israel_data["Year"].astype(str)
    ##get plot
    fig = px.scatter(israel_data,x  = israel_data['DayOfYear'],y = israel_data['Temp'],color = "Year" )

    fig.update_xaxes(title = "Day Of Year 1-365")
    fig.update_yaxes(title = "Average Daily Temprature in Celsius")

    fig.show()

    mon_data = full_data.groupby(["Month"])["Temp"].std()
    fig_bar = px.bar(mon_data)
    fig_bar.update_layout(title = "monthly standard deviation from average temprature",xaxis_title = "Month",yaxis_title = "Temp std" )
    fig_bar.show()

    # Question 3 - Exploring differences between countries
    country_mon_data = full_data.groupby(['Country','Month']).Temp.agg([np.mean,np.std])


    new_fig = px.line(country_mon_data,x=country_mon_data.index.get_level_values('Month'),y='mean',error_y='std',
                      color=country_mon_data.index.get_level_values('Country'))
    new_fig.update_layout(title = "Temp Mean by country + standard deviation",xaxis_title = "Month",yaxis_title = "Temprature mean")

    new_fig.show()



    # Question 4 - Fitting model for different values of `k`
    israel_samples = israel_data['DayOfYear']
    israel_response = israel_data['Temp']
    train_x,train_y,test_x,test_y = split_train_test(israel_samples,israel_response, 0.75)
    loss_arr = []
    for i in range(1,11):
        polyfit = PolynomialFitting(i)
        polyfit.fit(train_x,train_y)
        x = polyfit.loss(test_x,test_y)
        loss_arr.append(x)
    rounded_loss_arr = np.round(loss_arr,2)
    bar_plot = px.bar(x = [range(1,11)],y = rounded_loss_arr)
    bar_plot.update_xaxes(title = "polynomial fit degree")
    bar_plot.update_yaxes(title = "loss value")
    bar_plot.show()


    # Question 5 - Evaluating fitted model on different countries
    # +1 because we get list indice and we want degree
    min_loss = np.argmin(rounded_loss_arr)+1
    print(min_loss)
    min_fit = PolynomialFitting(min_loss)
    min_fit.fit(israel_samples,israel_response)
    country_labels = ['Israel','Jordan',
                      'The Netherlands','South Africa']
    country_loss =[]
    for country in country_labels:
        country_data = full_data[full_data['Country'] == country]
        country_samples = country_data['DayOfYear']
        country_response = country_data['Temp']
        loss = min_fit.loss(country_samples,country_response)
        country_loss.append(loss)
    country_bar_plot = px.bar(x=country_labels, y=country_loss)
    country_bar_plot.update_xaxes(title="countries")
    country_bar_plot.update_yaxes(title="loss value")
    country_bar_plot.show()