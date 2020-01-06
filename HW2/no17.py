# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 18:33:04 2020

@author: 15608
"""
import numpy as np

def get_theta(x):#由输入的x生成假设空间的所有theta的序列
    n=len(x)
    l1=sorted(x)
    theta=np.zeros(n)
    for i in range(n-1):
        theta[i]=(l1[i]+l1[i+1])/2
    theta[-1]=1
    return theta

def h(x,theta,s=1):
    return s*np.sign(x-theta)

def cal_ein(hx,y):
    return np.sum(hx!=y)/len(y)

def cal_eout(s, theta):
    return 0.5+0.3*s*(np.abs(theta)-1)

def experinment(times=2000):
    results = []
    results2 = []
    for i in range(times):
        X = np.random.uniform(-1,1,20)
        y = np.sign(X)
        index = np.random.choice(a=len(y),size=int(0.2*len(y)),replace=False)
        y[index] = -y[index] # y中有0.2的噪声
        thetas = get_theta(X)
        # s=+1
        e_ins = []
        e_outs = []
        for theta in thetas:
            e_ins.append(cal_ein(h(X, theta), y))
            e_outs.append(cal_eout(1, theta))
        # s=-1
        for theta in thetas:
            e_ins.append(cal_ein(h(X, theta, s=-1), y))
            e_outs.append(cal_eout(-1, theta))
            
        results.append(np.min(e_ins))
        results2.append(np.min(e_outs))
    return np.average(results), np.average(results2)

print(experinment())
        
    
    

