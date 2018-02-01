# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:11:00 2018

@author: 56999
"""

import pandas as pd
import numpy as np
import auto_bin_woe
from sklearn.cross_validation import train_test_split 

#read_file_path=r'C:\Users\56999\Desktop\LLD\project\hulu_install_mod\02 characteristic_work\hulu_woe_final.csv'
#
#df = pd.read_csv(read_file_path,encoding = "GBK")
#x_columns = [x for x in df.columns if x not in 'gbflag']
X = auto_bin_woe.X_woe_selected
y = auto_bin_woe.y

#df.describe()

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train,y_train)
pred = gbdt.predict(X_test)
pd.crosstab(y_test,pred)

#算法评估指标
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, digits=4))

#交叉验证（数据样本少，可以使用交叉验证方法）
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
kf = KFold(n_splits = 10)
scores = []
for train,test in kf.split(X):
    train_X,test_X,train_y,test_y = X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test]
    gbdt =  GradientBoostingClassifier(max_depth=4,max_features=5,n_estimators=100)
    gbdt.fit(train_X,train_y)
    prediced = gbdt.predict(test_X)
    print(accuracy_score(test_y,prediced))
    scores.append(accuracy_score(test_y,prediced))   
##交叉验证后的平均得分
np.mean(scores)

#自动调参
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
gbdt = GradientBoostingClassifier()
cross_validation = StratifiedKFold(y,n_folds = 10)
parameter_grid = {'max_depth':[2,3,4,5],
                  'max_features':[1,3,5,7,9],
                  'n_estimators':[10,30,50,70,90,100]}
grid_search = GridSearchCV(gbdt,param_grid = parameter_grid,cv =cross_validation,
                           scoring = 'accuracy')
grid_search.fit(X,y)
grid_search.best_score_
grid_search.best_params_
grid_search.estimator
