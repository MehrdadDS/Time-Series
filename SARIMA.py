# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:53:48 2022

@author: Mehrdad.Dadgar
"""


# IMPORT LIBRARIES ------------------------------------------------------------
import pandas as pd
import mysql.connector as connection
#from fbprophet import Prophet
from datetime import datetime,timedelta
import numpy as np
import statsmodels.api as sms
import matplotlib.pyplot as plt
import statsmodels

import warnings                                  # do not disturbe mode
warnings.filterwarnings('ignore')
import seaborn as sns  

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

# Importing everything from forecasting quality metrics
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error






# Functions ---------------------------------------------------------------------
def Make_Weekly_data(df):
    r = df.groupby(['Year','Week Number','Terminal']).sum()
    r = pd.DataFrame(r).reset_index()
    return r



def future_weeks(df,n):
    df = df.sort_values(['Year','Week Number'])
    start_w =df['Week Number'].iloc[-1]+1
    finish_w = start_w + 52
    finish_y = df['Year'].iloc[-1]
    arr_w = np.arange(start_w,finish_w,)
    arr_w[-start_w+1:] = arr_w[-start_w+1:] - 52
    arr_y = arr_w.copy()
    arr_y[:] = finish_y
    arr_y[-start_w+1:] = finish_y + 1
    future_df  = pd.DataFrame([])
    future_df['Year'] = arr_y
    future_df['Week Number'] = arr_w
    return(future_df)

# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    if y_true !=0:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        return 0


def mean_absolute_error(y_true, y_pred): 
        return np.mean(np.abs(y_true - y_pred))


    
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
        
def optimizeSARIMA(y, parameters_list, d, D, s):
    """Return dataframe with parameters and corresponding AIC
        
        y - time series
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """

    results = []
    best_aic = float("inf")
    
    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(y, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
    
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table    
    
  





def plotSARIMA(series, model, n_steps):
    """Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future    
    """
    
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.sarima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['sarima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)      

# Queries ---------------------------------------------------------------------
mydb = connection.connect(host="localhost", database = 'forecasting_schema',user="root", passwd="123456",use_pure=True)

query_daily = "Select * from fmr;"
df_daily_import = pd.read_sql(query_daily,mydb)

terminal_query = "Select * from terminal_list;"
Terminal_names_import = pd.read_sql(terminal_query,mydb)

query_holidays_by_day = "Select * from holidays_by_day"
Holidays_import = pd.read_sql(query_holidays_by_day,mydb)




# Data prepration ---------------------------------------------------------------------
##      Terminals
Terminal_names = Terminal_names_import[['Terminal','Division','Terminal Name']]
##      Dates table
Holidays = Holidays_import[['Date','Year','Week Number']]
Holidays.columns = ['ds','Year','Week']
Holidays = Holidays.drop_duplicates(['ds','Year','Week'])
Holidays.set_index('ds',inplace=True)

##      Temporary variables
Terminal_list = sorted(df_daily_import['Terminal'].unique())
forecast_total = pd.DataFrame([])

df_daily = df_daily_import[['CalendarDate','Year','Week Number','Terminal','Total Del Stops']]
df_daily.columns = ['ds','Year','Week Number','Terminal','y']


df_weekly = Make_Weekly_data(df_daily[df_daily['ds']<'2022-01-26'])


Terminal_list = [12,174]
for terminal in Terminal_list :
    #terminal=62
    data_d = df_daily[['ds','Terminal','y']]
    data_d = data_d[data_d['Terminal']==terminal]
    data_d = data_d [['ds','y']]
    data_d.set_index('ds',inplace=True,drop=True)
    data_d = data_d['2019-01-01':]
    
    start_date_actual = '2019-01-01'
    end_date_actual = '2022-01-30'
    
    data_d = data_d[start_date_actual:end_date_actual]
    data_range = pd.date_range(start_date_actual,end_date_actual)
    data_d = data_d.reindex(data_range,fill_value=0)
    
    
    
    
    
    
    
    plt.figure(figsize=(18, 6))
    plt.plot(data_d)
    plt.title('Ads watched (hourly data)')
    plt.grid(True)
    plt.show()
    
    tsplot(data_d['y'], lags=30)


    # The seasonal difference
    ads_diff = data_d - data_d.shift(7)
    ads_diff = ads_diff[7:]
    ads_diff.plot()
    tsplot(ads_diff['y'], lags=30)
    
    
    # setting initial values and some bounds for them
    ps = range(0, 2)
    d=0 
    qs = range(0, 2)
    Ps = range(0, 1)
    D=1 
    Qs = range(0, 1)
    s = 7 # season length is still 7
    
    # creating list with all the possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)

    

    best_model=sm.tsa.statespace.SARIMAX(data_d.y, order=(1, 0, 1), 
                                            seasonal_order=(1, 1, 1, 7)).fit(disp=-1)
    print(best_model.summary())

    tsplot(best_model.resid[7+1:], lags=60)


    plotSARIMA(data_d, best_model, 10)


    %%time
    #warnings.filterwarnings("ignore") 
    #result_table = optimizeSARIMA(data_d['y'], parameters_list, d, D, s)
    
    
    
    
    """Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future    
    """
    
    # adding model values
    data = data_d.copy()
    data.columns = ['actual']
    data['sarima_model'] = best_model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = best_model.predict(start = data.shape[0], end = data.shape[0]+5)
    forecast = data.sarima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_error(data['actual'][s+d:], data['sarima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)   
    
    
    
    
    
    best_model.predict(start = 1126, end = 700+1126).plot()
    
    
    (end_date_actual).dtype()
    
    
    
    
    from statsmodels.tsa.arima.model import ARIMA

    m = ARIMA(data['actual'],order=(7,0,7))
    model_fit = m.fit()    
    model_fit.predict(start = 1126, end = 100+1126).plot()
    
    
    
   dw = df_weekly[df_weekly['Terminal']==terminal] ['y']
   tsplot(dw.diff(1).dropna(), lags=60)
   m = ARIMA(dw,order=(52,1,52))
   model_fit = m.fit()    
   model_fit.predict(start = dw.shape[0], end = 52+dw.shape[0]).plot()
