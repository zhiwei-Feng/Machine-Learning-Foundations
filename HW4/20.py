# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:27:36 2020

@author: 15608
"""

import numpy as np
import pandas as pd

data_train = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_train.dat", header=None, delimiter=" ").to_numpy()
data_test = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_test.dat", header=None, delimiter=" ").to_numpy()

X_train = data_train[:, :2]
y_train = data_train[:, 2]
X_test = data_test[:, :2]
y_test = data_test[:, 2]

class RidgeRegression:
    
    def __init__(self, lambd=1):
        self.lambd = lambd
    
    def train_val(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        part = X.shape[0]//5
        errs = []
        # k fold
        for k in range(5):
            index = [i for i in range(k*part,(k+1)*part)]
            X_train = np.delete(X,index,axis=0)
            y_train = np.delete(y,index)
            X_val = X[index, :]
            y_val = y[index]
            self.w = np.linalg.inv(self.lambd*np.identity(X_train.shape[1])+X_train.T@X_train)@X_train.T@y_train
            
            y_hat = self.predict(X_val)
            err = np.sum(y_hat!=y_val)/y_val.size
            errs.append(err)
        
        return np.sum(errs)/5
    
    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.inv(self.lambd*np.identity(X.shape[1])+X.T@X)@X.T@y
        
        
    
    def predict(self, X):
        return np.sign(X@self.w)
    
lambdas = [10**x for x in range(2, -11, -1)]
E_cvs = []
for lambd in lambdas: 
    model = RidgeRegression(lambd=lambd)
    E_cv = model.train_val(X_train, y_train)
    E_cvs.append(E_cv)

min_lambda_index = np.argmin(E_cvs)
min_lambda = lambdas[min_lambda_index]

model = RidgeRegression(lambd=min_lambda)
model.train(X_train, y_train)

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
E_in = np.sum(y_train!=model.predict(X_train))/y_train.size
E_out = np.sum(y_test!=model.predict(X_test))/y_test.size

print(E_in, E_out)
