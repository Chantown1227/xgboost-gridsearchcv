# -*- coding = utf-8 -*-
# @time:2020/9/22 22:55
# Author:TC
# @File:xgb_test.py
# @Software:PyCharm

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123456)


# # 定义模型训练参数
# params = {
#     "objective": "binary:logistic",
#     "booster": "gbtree",
#     "max_depth": 3
#          }
# # 训练轮数
# num_round = 5
#
# # 训练过程中实时输出评估结果
# watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
#
# # 模型训练
# model = xgb.train(params, xgb_train, num_round, watchlist)


# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax', #多分类问题
    'num_class': 3,               #类别数，与 multisoftmax 并用
    'gamma': 0.1,                 #用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,               #构建树的深度，越大越容易过拟合
    'lambda': 2,                  #控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,             #随机采样训练样本
    'colsample_bytree': 0.75,     #生成树时进行的列采样
    'min_child_weight': 3,        #min_child_weight就是叶子上的最小样本数
    'silent': 1,                  #设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,                   #如同学习率
    'seed': 1,
    'nthread': 4,                 #cpu 线程数
}

plst=list(params.items())

#先保存到XGBoost二进制文件中将使加载速度更快，然后再加载进来
dtrain=xgb.DMatrix(X_train,y_train)
num_rounds=500
model=xgb.train(plst,dtrain,num_rounds)

dtest=xgb.DMatrix(X_test)
y_pred=model.predict(dtest)

accuracy=accuracy_score(y_test,y_pred)
print('accuracy:%.2f%%'%(accuracy*100.0))

#显示重要特征
plot_importance(model)
plt.show()
