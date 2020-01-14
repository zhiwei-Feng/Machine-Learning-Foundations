# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:02:45 2020

@author: 15608
"""

import numpy as np

def f(x):
    x = x**2
    w = np.array([-0.6,1,1])
    return np.sign(x@w)

def experi():
    X = np.random.uniform(-1,1,size=(1000,2))
    X = np.c_[np.ones(X.shape[0]),X]
    y = f(X)
    noise = np.random.choice(1000, 100)
    y[noise] = -y[noise]
    
    W_lin = np.linalg.inv(X.T@X)@X.T@y
    
    y_hat = np.sign(X@W_lin)
    
    E_in = np.sum(y!=y_hat)
    return E_in

E_ins = []
for i in range(1000):
    E_ins.append(experi())
print(np.average(E_ins)/1000)
        
        
        
        