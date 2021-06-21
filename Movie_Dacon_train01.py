# Ignore the warnings
from typing import cast
import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import numpy as np
from itertools import product 
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

# Evaluation metrics
# for regression
from sklearn.metrics import mean_squared_log_error, mean_squared_error,  r2_score, mean_absolute_error
# for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import SimpleRNN, LSTM, GRU

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

os.chdir('C:/Users/User/OneDrive/바탕 화면/dacon/movies')

# 데이터 불러오기
train = pd.read_csv('movies_train.csv')
test = pd.read_csv('movies_test.csv')
submission = pd.read_csv('submission.csv')

#EDA
train.head()
#title : 영화의 제목
#distributor : 배급사
#genre : 장르
#release_time : 개봉일
#time : 상영시간(분)
#screening_rat : 상영등급
#director : 감독이름
#dir_prev_bfnum : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화에서의 평균 관객수(단 관객수가 알려지지 않은 영화 제외)
#dir_prev_num : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화의 개수(단 관객수가 알려지지 않은 영화 제외)
#num_staff : 스텝수
#num_actor : 주연배우수
#box_off_num : 관객수

train[['genre', 'box_off_num']].groupby('genre').mean().sort_values('box_off_num')

#상관관계
train.corr()

sns.heatmap(train.corr(), annot=True)

#결측치
test.isna().sum()
#결측값 추출 
train[train['dir_prev_bfnum'].isna()]

#결측값 중에서  dir_prev_num값이 전체 0인지 확인
train[train['dir_prev_bfnum'].isna()]['dir_prev_num'].sum()

#결측값에 0 값을 넣어주기
train['dir_prev_bfnum'].fillna(0, inplace=True)
test['dir_prev_bfnum'].fillna(0, inplace=True)


#변수 선택 및 모델 구축
#Feature Engineering % Initial Modeling
model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)
#n_estimators : 순차적으로 진행하는 모델은 100개를 만들겠다.
features = ['time', 'dir_prev_num', 'num_staff', 'num_actor']
target = ['box_off_num']
X_train, X_test, y_train = train[features], test[features], train[target]

#5. 모델 학습 및 검증
#Model Tuning & Evaluation
#a. lightGBM (base model)

model.fit(X_train, y_train)
singleLGBM = submission.copy()
singleLGBM.head()
singleLGBM['box_off_num'] = model.predict(X_test)
singleLGBM.to_csv('singleLGBM.csv', index = False)

#b. k-fold lightGBM (k-fold model)
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=777) #shuffle 은 한번 섞는다. (시계열에서는 섞지 않는다)
for train_idx, val_idx in k_fold.split(X_train):
    print(len(train_idx), len(val_idx))
    break

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)

models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_t = X_train.iloc[train_idx]
    y_t = y_train.iloc[train_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    models.append(model.fit(x_t, y_t, eval_set=(x_val, y_val), early_stopping_rounds=100, verbose = 100))
models

preds = []
for model in models:
    preds.append(model.predict(X_test))
len(preds)

kfoldLightGBM = submission.copy()
kfoldLightGBM['box_off_num'] = np.mean(preds, axis = 0)
kfoldLightGBM.head()
kfoldLightGBM.to_csv('kfoldLightGBM.csv', index = False)

#c. feature engineering (fe)

features
train.columns
train.genre

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train['genre'] = le.fit_transform(train['genre'])
train['genre']

test['genre'] = le.transform(test['genre'])

features = ['time', 'dir_prev_num', 'num_staff', 'num_actor', 'dir_prev_bfnum', 'genre']

X_train, X_test, y_train = train[features], test[features], train[target]

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)

models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_t = X_train.iloc[train_idx]
    y_t = y_train.iloc[train_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    
    models.append(model.fit(x_t, y_t, eval_set=(x_val, y_val), early_stopping_rounds=100, verbose = 100))

X_test.head()

preds = []
for model in models:
    preds.append(model.predict(X_test))
len(preds)

feLightGBM = submission.copy()
feLightGBM['box_off_num'] = np.mean(preds, axis = 0)
feLightGBM.to_csv('feLightGBM.csv', index = False)


#d. grid search (hyperparameter tuning)
from sklearn.model_selection import GridSearchCV

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000)


params = {
    'learning_rate': [0.1, 0.01, 0.003],
    'min_child_samples': [20, 30]}

gs = GridSearchCV(estimator=model,
            param_grid=params,
            scoring='neg_mean_squared_error',
            cv = k_fold)

gs.fit(X_train, y_train)
gs.best_params_

model = lgb.LGBMRegressor(random_state=777, n_estimators=1000, learning_rate= 0.003, min_child_samples=30)

models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_t = X_train.iloc[train_idx]
    y_t = y_train.iloc[train_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    
    models.append(model.fit(x_t, y_t, eval_set=(x_val, y_val), early_stopping_rounds=100, verbose = 100))

preds = []
for model in models:
    preds.append(model.predict(X_test))


gs.best_score_
gslgbm = submission.copy()
gslgbm['box_off_num'] =  np.mean(preds, axis = 0)
gslgbm.to_csv('gslgbm.csv', index = False)