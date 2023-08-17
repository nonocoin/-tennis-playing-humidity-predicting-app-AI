#Library
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("tennis_database.csv") #probability of playing tennis database
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data2 = data.apply(LabelEncoder().fit_transform)
c = data2.iloc[:,:1]

from sklearn import preprocessing

ohe = preprocessing.OneHotEncoder()

c=ohe.fit_transform(c).toarray()
result_outlook = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])

s = pd.concat([result_outlook,data.iloc[:,1:3]],axis = 1)
s = pd.concat([data2.iloc[:,-2:],s], axis = 1)

from sklearn.model_selection import train_test_split

s_y = s
s_y = s_y.drop("o",axis=1)

x_train, x_test,y_train,y_test = train_test_split(s_y,s.iloc[:,2:-4],test_size=0.33, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm
X =  np.append(arr = np.ones((14,1)).astype(int), values = s.iloc[:,:-1],axis=1)

X_l = s.iloc[:,[2,4,5]].values

model = sm.OLS(s.iloc[:,-1:],X_l)
r = model.fit()
#print(r.summary())

x_train = x_train.iloc[:,0:6]  #[:,[0,1,2,3,4,5]] mean the same thing x_train and x_test cannot have different slots!
x_test = x_test.iloc[:,[0,1,2,3,4,5]] #[:,0:6] 

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
#y_pred = sc.fit_transform(y_pred) #The standardscaler is unnecessary here, but you can shrink large numbers in different projects.
"""

print()
print("Is Tennis Playable? ? (AI)")
i = 0
while True:
    print(round(y_pred[i][0],3)) #prints the value
    i+=1
    if len(y_pred) == i:
        break

print("Is Tennis Playable? 0-1 (REAL)")
print(y_test)

deger_oyun = y_pred
"""
print("AI Game Playing Status ")
print(y_pred)
print()
print("Real Game Playing Status ")
print(y_test)
#///////////////////////////////////////////////////////////////////////////


x_train, x_test,y_train,y_test = train_test_split(s.iloc[:,:-1],s.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm
X =  np.append(arr = np.ones((14,1)).astype(int), values = s.iloc[:,:-1],axis=1)

X_l = s.iloc[:,[2,4,5]].values

model = sm.OLS(s.iloc[:,-1:],X_l)
r = model.fit()
#print(r.summary())

x_train = x_train.iloc[:,1:]  #[:,[0,1,2,3,4,5]]mean the same thing x_train and x_test cannot have different slots!
x_test = x_test.iloc[:,1:] #[:,0:6] 

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
#y_pred = sc.fit_transform(y_pred) #standard scaler is unnecessary here, but you can reduce large numbers in different projects.
print()
print("Humidity Value (AI)")
print(y_pred) #prints the value
print("Humidity Value (REAL)")
print(y_test)



