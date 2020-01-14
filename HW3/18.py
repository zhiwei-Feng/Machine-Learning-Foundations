# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:25:24 2020

@author: 15608
"""

import numpy as np
import pandas as pd
# prehandle
data = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat", header=None, delimiter=' ')
data = data.dropna(axis=1)