
#Here I developed a WalkForward Validation algorithm for timeseries, you can easily use lines 28 on in your model
# This model has been applied to Holt Model but you can use it in other models as well.
#Please let me know if you have any question
#mehrdad.dadgar@purolator.com



# WALK-Forward Validation
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing




df = pd.read_csv(r'C:\Users\Dadgar\Desktop\Code\airline_passengers.csv',index_col='Month',parse_dates=True)
df.head()
df.index.freq='MS'



# Assume the forecast horizen we care about si 12
# Validate over 10 steps
h = 12
steps = 10
Ntest = len(df) - h - steps + 1
# Configuration hyperparameters to try
trend_type_list = ['add','mul']
seasonal_type_list = ['add','mul']
damped_trend_list = [True,False]
init_method_list = ['estimated','heuristic','legacy-heuristic']
use_boxcox_list= [True,False,0]




def walkforward(trend_type,seasonal_type,damped_trend,
                                    init_method,use_boxcox,debug=False):
    
    errors=[]
    seen_last=False
    steps_completed = 0

    for end_of_train in range(Ntest,len(df)-h+1):
        train = df.iloc[:end_of_train]
        test = df.iloc[end_of_train:end_of_train+h]

        if test.index[-1] ==df.index[-1]:
            seen_last = True
        
        steps_completed +=1

        hw = ExponentialSmoothing(train['Passengers'],initialization_method=init_method,
        trend = trend_type,damped_trend=damped_trend,seasonal = seasonal_type,seasonal_periods=12,use_boxcox=use_boxcox)
        res_hw = hw.fit()
        fcast = res_hw.forecast(h)
        error = mean_squared_error(test['Passengers'],fcast)
        errors.append(error)

        if debug:
            print("seen_last:",seen_last)
            print("steps completed:",steps_completed)

    return np.mean(errors)


# test our function
walkforward('add','add',False,'legacy-heuristic',0,debug=True) 


# Iterate through all possible options (Grid search)
tuple_of_option_lists = (trend_type_list,seasonal_type_list,
                            damped_trend_list,init_method_list,use_boxcox_list)

for x in itertools.product(*tuple_of_option_lists):
    print(x)


best_score = float('inf')
best_options = None
for x in itertools.product(*tuple_of_option_lists):
    score = walkforward(*x)

    if score<best_score :
        print("Best score so far :",score)
        best_score = score
        best_options = x


print("best score:",best_score)

trend_type ,seasonal_type,damped_trend,init_method,useboxcox = best_options
print("trend_type",trend_type)
print("seasonal_type",seasonal_type)
print("damped_trend",damped_trend)
print("init_method",init_method)
print("use_boxcox",useboxcox)