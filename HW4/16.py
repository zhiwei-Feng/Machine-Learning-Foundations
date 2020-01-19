# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:36:12 2020

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

X_train, X_val, y_train, y_val = X_train[:120,:], X_train[120:,:], y_train[:120], y_train[120:]

class RidgeRegression:
    
    def __init__(self, lambd=1):
        self.lambd = lambd
    
    def train(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.inv(self.lambd*np.identity(X.shape[1])+X.T@X)@X.T@y
    
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return np.sign(X@self.w)
    
lambdas = [10**x for x in range(2, -11, -1)]
E_trains = []
E_vals = []
E_outs = []
for lambd in lambdas: 
    model = RidgeRegression(lambd=lambd)
    model.train(X_train, y_train)
    
    y_hat = model.predict(X_train)
    E_train = np.sum(y_hat!=y_train)/y_train.size
    E_trains.append(E_train)
    
    y_hat = model.predict(X_val)
    E_val = np.sum(y_hat!=y_val)/y_val.size
    E_vals.append(E_val)
    
    y_hat = model.predict(X_test)
    E_out = np.sum(y_hat!=y_test)/y_test.size
    E_outs.append(E_out)

min_lambda_index = np.argmin(E_trains)
print(lambdas[min_lambda_index], E_trains[min_lambda_index], E_vals[min_lambda_index], E_outs[min_lambda_index])