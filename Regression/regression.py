# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:09:24 2018

@author: Ray Liu
"""

import pandas as pd
import quandl as qd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = qd.get('WIKI/GOOGL')


df = df [['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']* 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']* 100

df = df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace= True)

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])


X = preprocessing.scale(X)
y= np.array(df['label'])

X_train, X_test, y_train, y_test= cross_validation.train_test_split(X,y,test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)