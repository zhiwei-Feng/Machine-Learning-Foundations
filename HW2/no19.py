# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:28:42 2020

@author: 15608
"""

import pandas as pd
import numpy as np

# train
data_train = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw2_train.dat",header=None,delimiter=" ")
data_train = data_train.dropna(axis=1).to_numpy()
X_train = data_train[:,:9]
y_train = data_train[:,9]

# test
data_test = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw2_test.dat",header=None,delimiter=" ")
data_test = data_test.dropna(axis=1).to_numpy()
X_test = data_test[:,:9]
y_test = data_test[:,9]
## 数据清洗完毕！

class DecisionStump:
    
    def __init__(self):
        self.best_thetas=[]
        self.e_ins=[]
        self.e_outs=[]
        self.best_s=[]

        
    def get_theta(self,x):#由输入的x生成假设空间的所有theta的序列
        n=len(x)
        l1=sorted(x)
        theta=np.zeros(n)
        for i in range(n-1):
            theta[i]=(l1[i]+l1[i+1])/2
        theta[-1]=1
        return theta
    
    def cal_ein(self,hx,y):
        return np.sum(hx!=y)/len(y)

    def cal_eout(self,s, theta):
        return 0.5+0.3*s*(np.abs(theta)-1)
    
    def h(self,x,theta,s=1):
        return s*np.sign(x-theta)
    
    def train(self,X,y):
        for i in range(X.shape[1]):
            Xi = X[:,i]
            thetas = self.get_theta(Xi)
            # s=+1
            e_ins = []
            for theta in thetas:
                e_ins.append(self.cal_ein(self.h(Xi, theta), y))

            # s=-1
            for theta in thetas:
                e_ins.append(self.cal_ein(self.h(Xi, theta, s=-1), y))
            self.e_ins.append(np.min(e_ins))
            index=np.argmin(e_ins)
            if index>=len(thetas):
                self.best_s.append(-1)
                index=np.argmin(e_ins)%len(thetas)
            else:
                self.best_s.append(1)
            self.best_thetas.append(thetas[index])
                
    def predict_err(self, X_test, y_test):
        e_outs = []
        index = np.argmin(self.e_ins)
        theta = self.best_thetas[index]
        s = self.best_s[index]
        
        for j in range(X_test.shape[1]):
            X = X_test[:,j]
            y_pred = self.h(X, theta, s)
            e_out = np.sum(y_pred!=y_test)/y_test.size
            e_outs.append(e_out)
            
        return e_outs
            
            
        

model = DecisionStump()
model.train(X_train,y_train)
print(np.min(model.e_ins))
print(np.min(model.predict_err(X_test, y_test)))
            
        