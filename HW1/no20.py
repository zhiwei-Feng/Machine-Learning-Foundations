# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:34:45 2019

@author: 15608
"""

import numpy as np
import pandas as pd
import copy



class Perceptron():
    
    def __init__(self, X, y, eta, maxcount=1000):
        self.X = np.c_[np.ones(len(y)),X];
        self.y = y
        self.eta = eta
        self.temp_w = np.zeros(self.X.shape[1])
        self.best_w = np.zeros(self.X.shape[1])
        self.maxcount=maxcount
    
    def train(self):
        count=0
        while count<self.maxcount:
            self.best_error = sum(np.sign(self.X.dot(self.best_w))!=self.y)
            for i in range(len(self.y)):
                x = self.X[i, :]
                y = self.y[i]
                if y * self.temp_w.dot(x) <= 0:
                    self.temp_w = self.temp_w + self.eta*y*x
                    cur_error = sum(np.sign(self.X.dot(self.temp_w))!=self.y)
                    if cur_error<self.best_error:
                        self.best_w = copy.copy(self.temp_w)
                        self.best_error = cur_error
                    count+=1
    
    def predict(self, X):
        return np.sign(X.dot(self.best_w))
    

data = pd.read_csv('hw1_18_train.dat',header=None,delimiter=' ')
test = pd.read_csv('hw1_18_test.dat',header=None,delimiter=' ')
data = data.to_numpy()
test = test.to_numpy()

errors = []
for i in range(100):
    np.random.seed(i)
    np.random.shuffle(data)
    X = data[:, :4]
    y = data[:, 4]
    X_test = test[:, :4]
    y_test = test[:, 4]
    X_test = np.c_[np.ones(len(y_test)), X_test]
    model = Perceptron(X, y, 1, 100)
    model.train()
    y_pred = model.predict(X_test)
    error_rate = sum(y_test!=y_pred)/len(y_test)
    errors.append(error_rate)

print(np.sum(errors)/len(errors))