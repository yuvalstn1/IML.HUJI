import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
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


    samples = temp_data.drop('Temp', axis = 1)
    resp = temp_data['Temp']
    return samples,resp




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp_samples,response = load_data('../datasets/City_Temperature.csv')
    full_data = pd.concat([temp_samples,response],axis=1)
    # Question 2 - Exploring data for specific country
    israel_data = full_data[full_data['Country'] == 'Israel']
    # fig = px.scatter(x = israel_data['DayOfYear'],y= israel_data['Temp'] ,
    #                                    title = "Average Temprature as function of Day of year")
    # fig.update_xaxes(title = "Day Of Year 1-365")
    # fig.update_yaxes(title = "Average Daily Temprature in Celsius")
    #
    # fig.show()

    mon_data = full_data.groupby(["Month"])["Temp"].std()
    fig_bar = px.bar(mon_data)
    fig_bar.show()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()