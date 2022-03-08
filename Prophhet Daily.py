# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:04:34 2022

@author: Mehrdad.Dadgar
"""

# IMPORT LIBRARIES ------------------------------------------------------------
import pandas as pd
import mysql.connector as connection
from fbprophet import Prophet
from datetime import datetime,timedelta
import numpy as np


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




#Terminal_list = [12,53,174,552,561,524,508,132,490,541,120]
# Main section -------------------------------------------------------------------------
for terminal in Terminal_list :
    #terminal=12
    data_d = df_daily[df_daily['Terminal']==terminal] #552
    data_d.drop(labels=['Terminal','Year','Week Number'],axis=1,inplace=True)
    data_d.reset_index(inplace=True,drop=True)
    #start_week = data_w["Week Number"].iloc[0]
    #start_year = data_w["Year"].iloc[0]
    #date = "{}-{}-1".format(start_year, start_week)
    #start_date = datetime.strptime(date, "%Y-%W-%w")+timedelta(days=0)
    #data_w['ds'] = pd.date_range(start = start_date,freq='W',periods = len(data_w.index))
    #end_date = data_w['ds'].iloc[-1]
    end_date_actual = '2022-01-30'
    data_d = data_d[data_d['ds']<end_date_actual]
        
    #------------------------------------------------------------------------------
    
    #yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False,interval_width=0.95,mcmc_samples = 500)
    m = Prophet()
    #m.add_seasonality(name='monthly',period=30.5,fourier_order=2)
    m.fit(data_d)
    future = m.make_future_dataframe(periods=365)
    future = future[future['ds']>=end_date_actual].reset_index(drop=True)
    
    #future.set_index('ds',inplace=True,drop=True)

    forecast = m.predict(future)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    #forecast['ds'] = pd.DatetimeIndex(forecast['ds'])
    #m.plot(forecast['yhat'])
    fn = forecast[['ds','yhat']]
    
    
    data_d.columns = ['ds','yhat']
    #forecast = forecast[['ds','yhat']]
    forecast = fn.append(data_d)
    forecast.set_index('ds',drop=True,inplace=True)
    forecast['Type']='Actual'
    forecast = forecast.sort_values(by='ds')
    forecast.loc[end_date_actual:,'Type'] = 'Forecast'
    forecast['Terminal']=terminal
    forecast.reset_index(inplace=True)
    forecast = pd.merge(forecast,Terminal_names,on='Terminal')
    forecast = pd.merge(forecast,Holidays,on='ds')
    forecast_agg = forecast.groupby(['Year','Week','Terminal','Type'])['yhat'].sum().reset_index()
    
        
    forecast_total = forecast_total.append(forecast_agg)
    print(terminal)
    

    
forecast_total.to_excel(r"C:\Users\mehrdad.dadgar\Desktop\WeeklyProphet.xlsx",sheet_name='Total')    
    




