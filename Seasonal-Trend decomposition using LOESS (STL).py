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

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.rc("figure", figsize=(30, 17))
plt.rc("font", size=13)


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





def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)
                
                
                
                

def STL_forecast(model,model_input,data):
    stlf = STLForecast(data, model,model_kwargs=model_input,period= 365,trend=367)
    stlf_res = stlf.fit()
    forecast = stlf_res.forecast(365)
    forecast=pd.DataFrame(forecast,columns=['y'])
    new_d = pd.concat([data_d,forecast])
    new_d['Type'] = "Forecast"
    new_d.loc[:data_d.shape[0],'Type'] = 'Actual'
    
    plt.title("Daily chart {}".format(terminal))
    plt.plot(data['2022-01-01':])
    plt.plot(forecast)
    plt.show()
    return new_d


def Weekly_plot(weekly_forecast):
    week=4
    actual_line = weekly_forecast[(weekly_forecast['Year']< 2022) | ((weekly_forecast['Year']==2022) & (weekly_forecast['Week']<=4)) ]
    forecast_line = weekly_forecast[~((weekly_forecast['Year']< 2022) | ((weekly_forecast['Year']==2022) & (weekly_forecast['Week']<=4))) ]
    plt.title("Weekly chart {}".format(terminal))
    plt.plot(actual_line['y'])
    plt.plot(forecast_line['y'],c='red')
    plt.show()






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


df_weekly = Make_Weekly_data(df_daily[df_daily['ds']<'2022-01-30'])


#Terminal_list = [12,174,508,62]

Daily_table=pd.DataFrame([])
Weekly_table = pd.DataFrame([])

for terminal in Terminal_list :
    #terminal=30
    data_d = df_daily[['ds','Terminal','y']]
    data_d = data_d[data_d['Terminal']==terminal]
    data_d = data_d [['ds','y']]
    data_d.set_index('ds',inplace=True,drop=True)
    
    data_d = data_d.reset_index().drop_duplicates(subset='ds', keep='last').set_index('ds')
    
    start_date_actual = data_d.index[0]
    end_date_actual = '2022-01-29'
    
    data_d = data_d[start_date_actual:end_date_actual]
    data_range = pd.date_range(start_date_actual,end_date_actual)
    data_d = data_d.reindex(data_range,fill_value=0)
    
    
    data_d.index.freq = data_d.index.inferred_freq  
    model_input = dict(order=(1, 0, 1),seasonal_order=(1, 1, 1, 7))
    new_table = STL_forecast(SARIMAX,model_input,data_d)
    new_table['Terminal'] = terminal
    new_table= new_table.reset_index()
    new_table.columns=['ds','y','Type','Terminal']
    daily_forecast= pd.merge(new_table,Holidays,on='ds')
    
    weekly_forecast = pd.DataFrame(daily_forecast.groupby(['Year','Week','Terminal','Type']).sum()).reset_index()
    Weekly_plot(weekly_forecast)
    
    Weekly_table = Weekly_table.append(weekly_forecast)
    Daily_table= Daily_table.append(daily_forecast)
    
    print('{}***********************************************'.format(terminal))


Weekly_table.to_excel(r"C:\Users\mehrdad.dadgar\Desktop\WeeklySARIMAX.xlsx",sheet_name='Weekly',index=False)    
Daily_table.to_excel(r"C:\Users\mehrdad.dadgar\Desktop\DailySARIMAX.xlsx",sheet_name='Daily',index=False) 

"""    
    from statsmodels.tsa.seasonal import STL

    stl = STL(data_d, seasonal=7)
    res = stl.fit()
    fig = res.plot()
    
    stl = STL(data_d, period=7, robust=True)
    res_robust = stl.fit()
    fig = res_robust.plot()
    res_non_robust = STL(data_d, period=7, robust=False).fit()
    add_stl_plot(fig, res_non_robust, ["Robust", "Non-robust"])


    fig = plt.figure(figsize=(30, 15))
    lines = plt.plot(res_robust.weights, marker="o", linestyle="none")
    ax = plt.gca()
    xlim = ax.set_xlim(data_d.index[0], data_d.index[-1])




    stl = STL(
        data_d, period=7, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True
    )
    res_deg_0 = stl.fit()
    fig = res_robust.plot()
    add_stl_plot(fig, res_deg_0, ["Degree 1", "Degree 0"])
    
    
    
    
    


    tsplot(data_d['y'], lags=30)
    # The seasonal difference
    ads_diff = data_d - data_d.shift(7)
    ads_diff = ads_diff[7:]
    ads_diff.plot()
    tsplot(ads_diff['y'], lags=30)
    
"""    
    
    
    
    
    
    






