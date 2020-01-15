# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:05:37 2020

@author: 15608
"""

import numpy as np
import pandas as pd
# prehandle
train_data = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat", header=None, delimiter=' ')
train_data = train_data.dropna(axis=1).to_numpy()
test_data = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat",header=None, delimiter=' ')
test_data = test_data.dropna(axis=1).to_numpy()

X_train = train_data[:, :20]
y_train = train_data[:, 20]
X_test = test_data[:, :20]
y_test = test_data[:, 20]

class LogisticRegression:
    
    def __init__(self, eta=0.001, times=2000):
        self.eta=eta
        self.times=times
        
    def gradient(self, X, y):
        gt=0;
        for i in range(X.shape[0]):
            xn = X[i,:]
            yn = y[i]
            gt+=yn*xn/(1+np.exp(yn*xn.dot(self.w_)))
        return gt*(-1/X.shape[0])
    
    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.w_ = np.zeros(X.shape[1])
        
        for i in range(self.times):
            vt = -self.gradient(X, y)
            self.w_ = self.w_ + self.eta*vt
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    def err(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        y_hat = self.sigmoid(X.dot(self.w_))
        y_hat[y_hat>=0.5] = 1
        y_hat[y_hat<0.5] = -1
        return np.sum(y!=y_hat)/y.size
        
model = LogisticRegression(eta=0.01)
model.train(X_train,y_train)
print("E_out: {}".format(model.err(X_test, y_test)))    