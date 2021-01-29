# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:28:06 2021

@author: rao
"""

import sys
import os
import datetime
import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import time
# import shap
import seaborn as sns
from My_Modules import MAPELoss
from My_Modules import atomic_properties
from sklearn import linear_model
from sklearn import tree

plt.close('all')
starttime = datetime.datetime.now()

data = pd.read_excel('Data_base.xlsx')
df_test = data.drop(columns=['alloy'])

x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]


x2 = df_test [['VEC','AR1','AR2','PE','Density','TC','MP','FI','SI','TI','M']]


min = [8,135,124,1.8235,7874,75.905,1181,737.14,1562.98,2957.4,0.6]
max = [10,140,125.679,1.91,8908,116.559,1850.4,762.47,1753.03,3395,2.22]
min = np.array(min)
max = np.array(max)

x2_normalization = (x2 - min)/(max-min)

x_combined = pd.concat ((x1, x2_normalization), axis=1)

y1 = df_test [['TEC']]

x = x2_normalization.values

y = y1.values

train_features, test_features, train_labels, test_labels = train_test_split(x, 
                                                                            y, 
                                                                            test_size=0.15,
                                                                            random_state=39)

regr = tree.DecisionTreeRegressor(criterion='friedman_mse',max_depth=20)
regr.fit(train_features, train_labels)
y_pred = regr.predict(test_features)
print("预测性能得分: %.2f" % regr.score(test_features, test_labels))
train_pred = regr.predict (train_features)

plt.figure()
sns.set()
sns.regplot(x=test_labels, y=y_pred, robust=True, color='g') 

ax = plt.subplot()

ax.scatter(x=train_labels, y=train_pred, c='black', alpha=0.6)  # 改变颜色
ax.scatter(x=test_labels, y=y_pred, c='red',alpha=0.6)
plt.savefig('Figures/01_23_DT_max_depth_20.png', format='png', dpi=300)

matplotlib=True 
# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)




plt.show()

endtime = datetime.datetime.now()
print (endtime - starttime)

 