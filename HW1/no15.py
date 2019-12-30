# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:17:52 2019

@author: 15608
"""
import numpy as np
import pandas as pd

data = pd.read_csv('hw1_15_train.dat',header=None,delimiter=' ')
data = data.to_numpy()
X = data[:, :4]
y = data[:, 4]

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
                    flag=False
                    count+=1
            if flag:
                break
        
        print(count)
    
    


model = Perceptron(X, y, 1)
model.train()