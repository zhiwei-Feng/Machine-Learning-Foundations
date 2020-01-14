# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:56:36 2020

@author: 15608
"""

import numpy as np
def f(x):
    x = x**2
    w = np.array([-0.6,1,1])
    return np.sign(x@w)

def quadratic(X):
    xx = X[:,1]*X[:,2]
    x1_2 = X[:, 1]**2
    x2_2 = X[:, 2]**2
    return np.c_[X, xx, x1_2, x2_2]
    

def experi():
    X = np.random.uniform(-1,1,size=(1000,2))
    X = np.c_[np.ones(X.shape[0]),X]
    y = f(X)
    X = quadratic(X)
    noise = np.random.choice(1000, 100)
    y[noise] = -y[noise]
    
    Wlin = np.linalg.pinv(X.T@X)@X.T@y
    return Wlin

ws = []
for i in range(1000):
    ws.append(experi())
ws = np.array(ws)
print(np.average(ws,axis=0))