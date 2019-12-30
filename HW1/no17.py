# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:19:55 2019

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
    

data = pd.read_csv('hw1_15_train.dat',header=None,delimiter=' ')
data = data.to_numpy()

counts=[]
for i in range(2000):
    np.random.seed(i)
    np.random.shuffle(data)
    X = data[:, :4]
    y = data[:, 4]
    model = Perceptron(X, y, 0.5)
    counts.append(model.train())

print(np.sum(counts)/len(counts))