# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 01:14:18 2018

@author: Ray Liu
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
import numpy as np
import sklearn
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

price = datasets.load_boston()
#Puts data into datafram
bosprice = pd.DataFrame(price.data)
bosprice.columns = price.feature_names
bosprice['PRICE'] = price.target
print(bosprice.describe())
X_train, X_test, Y_train, Y_test =  sklearn.cross_validation.train_test_split(bosprice.drop('PRICE',axis = 1), bosprice['PRICE'], test_size = 0.2)


clf = LinearRegression(n_jobs=-1)
clf.fit(X_train,Y_train)
with open('linearregressionprac.pickle','wb') as f:
       pickle.dump(clf,f)
      
##pickle_in = open('linearregressionprac.pickle','rb')
##clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, Y_test)

print('Accuracy: ', accuracy)

