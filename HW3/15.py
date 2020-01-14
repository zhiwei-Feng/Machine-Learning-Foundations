# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:16:19 2020

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

def h(x, w):
    return np.sign(x@w)

def experi():
    W = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
    X = np.random.uniform(-1,1,size=(1000,2))
    X = np.c_[np.ones(X.shape[0]),X]
    y = f(X)
    X = quadratic(X)
    noise = np.random.choice(1000, 100)
    y[noise] = -y[noise]
    
    y_hat = h(X,W)
    E_out = np.sum(y_hat!=y)/y.size
    return E_out

Eouts = []
for i in range(1000):
    Eouts.append(experi())
print(np.average(Eouts))
