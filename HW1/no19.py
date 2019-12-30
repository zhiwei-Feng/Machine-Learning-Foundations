# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:30:32 2019

@author: 15608
"""

import numpy as np
import pandas as pd



class Perceptron():
    
    def __init__(self, X, y, eta, maxcount=1000):
        self.X = np.c_[np.ones(len(y)),X];
        self.y = y
        self.eta = eta
        self.w = np.zeros(self.X.shape[1])
        self.maxcount=maxcount
    
    def train(self):
        count=0
        while count<self.maxcount:
            flag = True
            for i in range(len(self.y)):
                x = self.X[i, :]
                y = self.y[i]
                if y * self.w.dot(x) <= 0:
                    self.w += self.eta*y*x
                    count+=1
                    flag=False
            if flag:
                break
        
        return count
    
    def predict(self, X):
        return np.sign(X.dot(self.w))
    

data = pd.read_csv('hw1_18_train.dat',header=None,delimiter=' ')
test = pd.read_csv('hw1_18_test.dat',header=None,delimiter=' ')
data = data.to_numpy()
test = test.to_numpy()

errors = []
for i in range(2000):
    np.random.seed(i)
    np.random.shuffle(data)
    X = data[:, :4]
    y = data[:, 4]
    X_test = test[:, :4]
    y_test = test[:, 4]
    X_test = np.c_[np.ones(len(y_test)), X_test]
    model = Perceptron(X, y, 1, 50)
    model.train()
    y_pred = model.predict(X_test)
    error_rate = sum(y_test!=y_pred)/len(y_test)
    errors.append(error_rate)

print(np.sum(errors)/len(errors))