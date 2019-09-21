# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:06:43 2019

@author: Yadav
"""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
location1='C:\\Users\\Yadav\\Desktop\\IIT MADRAS SANGRAM\\DataSets\\Train.csv'
location2='C:\\Users\\Yadav\\Desktop\\IIT MADRAS SANGRAM\\DataSets\\Test.csv'
#d=pd.read_csv(location1)
#print(d)
Train=pd.read_csv(location1)
Test=pd.read_csv(location2)
X_Train=Train[['is_holiday','air_pollution_index','rain_p_h','humidity','clouds_all','weather_type']]
t1=pd.get_dummies(X_Train,columns=["is_holiday","weather_type"])
print(t1.columns)
#print(t1[['is_holiday_Christmas Day']])
x_train=t1[['is_holiday_Christmas Day', 'is_holiday_Columbus Day',
       'is_holiday_Independence Day', 'is_holiday_Labor Day',
       'is_holiday_Martin Luther King Jr Day', 'is_holiday_Memorial Day',
       'is_holiday_New Years Day', 'is_holiday_None', 'is_holiday_State Fair',
       'is_holiday_Thanksgiving Day', 'is_holiday_Veterans Day',
       'is_holiday_Washingtons Birthday', 'weather_type_Clear',
       'weather_type_Clouds', 'weather_type_Drizzle', 'weather_type_Fog',
       'weather_type_Haze', 'weather_type_Mist', 'weather_type_Rain',
       'weather_type_Smoke', 'weather_type_Snow', 
       'weather_type_Thunderstorm']]
Y_Train=Train[['traffic_volume']]
X_Test=Test[['is_holiday','air_pollution_index','rain_p_h','humidity','clouds_all','weather_type']]
t2=pd.get_dummies(X_Test,columns=["is_holiday","weather_type"])
x_test=t2[['is_holiday_Christmas Day', 'is_holiday_Columbus Day',
       'is_holiday_Independence Day', 'is_holiday_Labor Day',
       'is_holiday_Martin Luther King Jr Day', 'is_holiday_Memorial Day',
       'is_holiday_New Years Day', 'is_holiday_None', 'is_holiday_State Fair',
       'is_holiday_Thanksgiving Day', 'is_holiday_Veterans Day',
       'is_holiday_Washingtons Birthday', 'weather_type_Clear',
       'weather_type_Clouds', 'weather_type_Drizzle', 'weather_type_Fog',
       'weather_type_Haze', 'weather_type_Mist', 'weather_type_Rain',
       'weather_type_Smoke', 'weather_type_Snow',
       'weather_type_Thunderstorm']]
Model=tree.DecisionTreeRegressor()
Model2=RandomForestRegressor()
y_train=np.array(Y_Train)
Model.fit(x_train,Y_Train)
Model2.fit(x_train,y_train)
predicted_values=Model.predict(x_test)
predicted_values2=Model2.predict(x_test)
print("Decision tree regressor-----------------")
print(predicted_values)
print("Random forest regressor-----------------")
print(predicted_values2)
df=Test[['date_time']]
df1=Test[['date_time']]
df.insert(1,value=predicted_values,column='traffic_volume')
df1.insert(1,value=predicted_values2,column='traffic_volume')
loc1='C:\\Users\\Yadav\\Desktop\\IIT MADRAS SANGRAM\\DataSets\\myoutputforsangramyadavendra.csv'
loc2='C:\\Users\\Yadav\\Desktop\\IIT MADRAS SANGRAM\\DataSets\\output2.csv'
df.to_csv(loc1)
#df1.to_csv(loc2)