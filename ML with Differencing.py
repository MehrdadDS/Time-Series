# WALK-Forward Validation
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r'C:\Users\Dadgar\Desktop\Code\airline_passengers.csv',index_col='Month',parse_dates=True)
df.head()
df.index.freq='MS'

import pmdarima as pm

df['LogPassengers'] = np.log(df['Passengers'])
df['DiffLogPassengers'] = df['LogPassengers'].diff()
Ntest=12
train = df[:-Ntest]
test = df[-Ntest:]
train_idx = df.index<=train.index[-1]
test_idx = df.index>train.index[-1]


# Make supervised dataset
# let's see if we can use T past values to predict the next value

series = df['DiffLogPassengers'].to_numpy()[1:]

T=10
X=[]
Y=[]

for t in range(len(series)-T):
    x=series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1,T)
Y = np.array(Y)
N = len(X)

print("X.shape",X.shape,"Y.shape",Y.shape)

Xtrain, Ytrain = X[:-Ntest],Y[:-Ntest]
Xtest, Ytest = X[-Ntest:],Y[-Ntest:]

lr = LinearRegression()
lr.fit(Xtrain,Ytrain)
lr.score(Xtrain,Ytrain)
lr.score(Xtest,Ytest)


# Boolean index
train_idx = df.index<=train.index[-1]
test_idx =~train_idx

train_idx[:T+1] = False # first T+1 values are not predictable 


# Needed to compute un-differenced predictions
df['ShiftLogPassengers'] = df['LogPassengers'].shift(1)
prev = df['ShiftLogPassengers']

# Last-known train value
last_train = train.iloc[-1]['LogPassengers']

# 1-step forecast
df.loc[train_idx,'LR_1step_train'] = prev[train_idx] + lr.predict(Xtrain)
df.loc[test_idx, 'LR_1step_test'] = prev[test_idx] + lr.predict(Xtest)

# plot 1-step forecast
df[['LogPassengers','LR_1step_train','LR_1step_test']].plot(figsize=(15,5))
plt.show()

#Incremental Multi-Step Forecast
# multi-step forecast
multistep_prediction = []

# first test input
last_x = Xtest[0]

while len(multistep_prediction)<Ntest:
    p = lr.predict(last_x.reshape(1,-1))[0]

    # update the prediction list
    multistep_prediction.append(p)

    #make the new input
    last_x = np.roll(last_x,-1)
    last_x[-1] = p

# save multi-step forecast to dataframe
df.loc[test_idx,'LR_multistep'] = last_train + np.cumsum(multistep_prediction)

# plot 1-step and multi-step forecast
df[['LogPassengers','LR_multistep','LR_1step_test']].plot(figsize=(15,5))
plt.show()


# make multi-output supervised dataset
Tx = T
Ty = Ntest
X=[]
Y = []
for t in range(len(series) - Tx - Ty + 1) :
    x = series[t:t+Tx]
    X.append(x)
    y = series[t+Tx:t+Tx+Ty]
    Y.append(y)


X = np.array(X).reshape(-1,Tx)
Y = np.array(Y).reshape(-1,Ty)
N = len(X)
print("X.shape",X.shape,"Y.shape",Y.shape)

Xtrain_m,Ytrain_m = X[:-1],Y[:-1]
Xtest_m,Ytest_m = X[-1:],Y[-1:]

lr = LinearRegression()
lr.fit(Xtrain_m,Ytrain_m)
lr.score(Xtrain_m,Ytrain_m)

r2_score(lr.predict(Xtest_m).flatten(),Ytest_m.flatten())


#save multi_opiutput forecast to dataframe1
df.loc[test_idx,'LR_multioutput'] = last_train + np.cumsum(lr.predict(Xtest_m).flatten())

#plot all forecasts
cols = ['LogPassengers','LR_multistep','LR_1step_test','LR_multioutput']
df[cols].plot(figsize=(15,5))
plt.show()


test_log_pass = df.iloc[-Ntest:]['LogPassengers']
def one_step_and_multistep_forecast(model,name):
    model.fit(Xtrain,Ytrain)
    print("one-step forecast:",name)

    # print("Train R^2:",model.score(Xtrain,Ytrain))
    # print("Test R^2 (1-step):",model.score(Xtest,Ytest))

    # store 1-step forecast
    df.loc[train_idx,f'{name}_1step_train'] = prev[train_idx] + model.predict(Xtrain)
    df.loc[test_idx,f'{name}_1step_train'] = prev[test_idx] + model.predict(Xtest)

    #generate multi_step forecast
    multistep_predictions = []

    # first test input
    last_x = Xtest[0]

    while len(multistep_predictions)<Ntest:
        p = model.predict(last_x.reshape(1,-1))[0]

        #update the prediction list
        multistep_predictions.append(p)

        # make the new input
        last_x = np.roll(last_x,-1)
        last_x[-1] = p

        # store multi-step forecast
        df.loc[test_idx,f'{name}_multistep_test'] = last_train + np.cumsum(multistep_predictions)
        
        #MAPE of multi-step forecast
        mape = mean_absolute_percentage_error(test_log_pass,df.loc[test_idx,f'{name}_multistep_test'])
        print("Test  MAPE (multi-step) :",mape)

        # plot 1-step and multi-step forecast
        cols = [ 'LogPassengers', f'{name}_1step_train',f'{name}_1step_test',f'{name}_multistep_test']
        df[cols].plot(figsize = (15,5))

one_step_and_multistep_forecast(SVR(),"SVR")





















model = pm.auto_arima(train['LogPassengers'],trace=True,suppress_warning=True,stepwise=False,max_p=12,max_q=2,max_order=14,seasonal=False)
model.summary()

test_pred,confint = model.predict(n_periods = Ntest,return_conf_int=True)
train_pred = model.predict_in_sample(start=0,end=-1)

df.loc[train_idx,'AATrain'] = train_pred
df.loc[test_idx,'AATest'] = test_pred
df.iloc[13:][['LogPassengers','AATrain','AATest']].plot()
plt.show()


from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
plot_pacf(df['LogPassengers'].diff(1).dropna())
plt.show()



















df['1stdiff'] = df['Passengers'].diff()


df['LogPassengers'] = np.log(df['Passengers'])
df['1stLogPassengers'] = df['LogPassengers'].diff()
df[['LogPassengers','1stLogPassengers']].plot()
plt.show()


from statsmodels.tsa.arima.model import ARIMA

Ntest=20
train = df[:-N_test]
test = df[-N_test:]
train_idx = df.index<=train.index[-1]
test_idx = df.index>train.index[-1]

arima = ARIMA(train['Passengers'],order=(1,0,0))
arima_result = arima.fit()
df.loc[train_idx,'AR(1)'] = arima_result.predict(start=train.index[0],end=train.index[-1])
df[['Passengers','AR(1)']].plot()
plt.show()

prediction_result = arima_result.get_forecast(Ntest)
forecast = prediction_result.predicted_mean
df.loc[test_idx,'AR(1)'] = forecast

prediction_result.conf_int()

# Order (1,0,1)
arima = ARIMA(train['Passengers'],order=(1,0,1))
arima_result = arima.fit()
df.loc[train_idx,'AR(1,1)'] = arima_result.predict(start=train.index[0],end=train.index[-1])
prediction_result = arima_result.get_forecast(Ntest)
forecast = prediction_result.predicted_mean
df.loc[test_idx,'AR(1,1)'] = forecast
df[['Passengers','AR(1,1)']].plot()
plt.show()

# Order (1,0,1)
arima = ARIMA(train['Passengers'],order=(8,1,1))
arima_result = arima.fit()
df.loc[train_idx,'AR(8,1,1)'] = arima_result.predict(start=train.index[0],end=train.index[-1])
prediction_result = arima_result.get_forecast(Ntest)
forecast = prediction_result.predicted_mean
df.loc[test_idx,'AR(8,1,1)'] = forecast
df[['Passengers','AR(8,1,1)']].plot()
plt.show()



# Order (1,0,1)
df['Log1stDiff']  = df['LogPassengers'].diff()
df['Log1stDiff'].plot()
plt.show()


arima = ARIMA(train['LogPassengers'],order=(12,1,0))
arima_result = arima.fit()
df.loc[train_idx,'AR(811)'] = arima_result.predict(start=train.index[0],end=train.index[-1])
prediction_result = arima_result.get_forecast(Ntest)
forecast = prediction_result.predicted_mean
df.loc[test_idx,'AR(811)'] = forecast
df[['LogPassengers','AR(811)']].dropna().plot()
plt.show()


arima = ARIMA(train['LogPassengers'],order=(12,1,1))
arima_result = arima.fit()
df.loc[train_idx,'AR(811)'] = arima_result.predict(start=train.index[0],end=train.index[-1])
prediction_result = arima_result.get_forecast(Ntest)
forecast = prediction_result.predicted_mean
df.loc[test_idx,'AR(811)'] = forecast
df[['LogPassengers','AR(811)']].dropna().plot()
plt.show()