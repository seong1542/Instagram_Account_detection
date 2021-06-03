# -*- coding: utf-8 -*-
"""instagram_machineLearning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1It5DXUB-vVKjXhBY1XC1BUoRaFacrKuE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
filepath='/content/gdrive/My Drive/'
train = pd.read_csv(filepath+'instagram_detection.csv')
print(train.shape)

train.head()

train.info()  #어떤 속성으로 이루어져있는지, 결측치가 있는지

#상관관계 분석
data_corr = train.corr(method='pearson')
axes = sns.heatmap(data_corr, vmin=-1, vmax=1, cmap='BrBG')
axes.set_title('Correlation Heatmap Between Features')

y_train = train['#fake']
X_train = train.drop('#fake',axis=1)

# 학습-검증데이터 8:2비율로
from sklearn.model_selection import train_test_split
X_train,X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.2, random_state=0, shuffle=True)

print('X_train = ', X_train.shape)
print('X_valid = ', X_valid.shape)
print('y_train = ', y_train.shape)
print('y_valid = ', y_valid.shape)

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X_train, y_train)
train_score = rf.score(X_train, y_train)
valid_score = rf.score(X_valid, y_valid)
print('RandomForest Train_Accuracy : {}'.format(train_score))
print('RandomForest Valid_Accuracy : {}'.format(valid_score))
print()

from sklearn.ensemble import GradientBoostingClassifier
gb= GradientBoostingClassifier()
gb.fit(X_train, y_train)
train_score = gb.score(X_train, y_train)
valid_score = gb.score(X_valid, y_valid)
print('GradientBoosting Train_Accuracy : {}'.format(train_score))
print('GradientBoosting Valid_Accuracy : {}'.format(valid_score))
print()

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(penalty = 'l2',solver='newton-cg',random_state=0, max_iter=600)
lg.fit(X_train,y_train)
train_score = lg.score(X_train,y_train)
valid_score = lg.score(X_valid,y_valid)
print('Logistic Train_Accuracy : {}'.format(train_score))
print('Logistic Valid_Accuracy : {}'.format(valid_score))
print()

import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train,y_train)
train_score = xgb_model.score(X_train,y_train)
valid_score = xgb_model.score(X_valid,y_valid)
print('XGBoost Train_Accuracy : {}'.format(train_score))
print('XGBoost Valid_Accuracy : {}'.format(valid_score))
print()

import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
train_score =lgb_model.score(X_train,y_train)
valid_score= lgb_model.score(X_valid,y_valid)
print('LGBM Train_Accuracy : {}'.format(train_score))
print('LGBM Valid_Accuracy : {}'.format(valid_score))
print()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
train_score = dt.score(X_train,y_train)
valid_score=dt.score(X_valid,y_valid)
print('DecisionTree Train_Accuracy : {}'.format(train_score))
print('DecisionTree Valid_Accuracy : {}'.format(valid_score))

"""# 최대한 효율적인 방법 - RandomForest, GradientBoosting, LGBM"""

#가짜계정인가? -> 0 = 진짜계정, 1 = 가짜계정
p=[[1,19,4,14,0,0,0,0,0,0,3,3,0,0,0,0]]  # 학습하지 않은, 가짜 계정의 정보를 입력
rf_y = rf.predict(p)
lgb_y=lgb_model.predict(p)
gb_y=gb.predict(p)
print(rf_y)
print(lgb_y)
print(gb_y)    #가짜계정이라고 답변 내리는 것을 볼 수 있음.

#가짜계정인가? -> 0= 진짜계정, 1 = 가짜계정
x=[[0,9,0,3,0,0,0,0,1,4,265,350,0,0,0,0]]  # 학습하지 않은, 진짜 계정의 정보를 입력
rf_y = rf.predict(x)
lgb_y=lgb_model.predict(x)
gb_y=gb.predict(x)
print(rf_y)
print(lgb_y)
print(gb_y)    #진짜계정이라고 답변 내리는 것을 볼 수 있음.

