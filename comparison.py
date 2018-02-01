# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:55:53 2017
@author: Hanyee
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:17:20 2017
@author: Hanyee
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import confusion_matrix
#数据导入
df = pd.read_csv('C:/Users/Hanyee/Desktop/model_data.csv')
x_columns = [x for x in df.columns if x not in 'gbflag']
X = df[x_columns]
y = df['gbflag']
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

#逻辑回归
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
preds=lr.predict_proba(X_test)
prob = pd.DataFrame(lr.predict_proba(X_test),columns=['B','G'])
preds = prob['G']

#SVM 模型
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf',class_weight='balanced',probability = True)
svc.fit(X_train,y_train)
pred = svc.predict(X_test)
prob = pd.DataFrame(svc.predict_proba(X_test),columns=['B','G'])
preds = prob['G']

#RadomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100 ,min_samples_split = 60 ,
max_depth = 3,max_features = 4,class_weight= 'balanced')
rf.fit(X_train, y_train)#进行模型的训练  
pred = rf.predict(X_test) 
prob = pd.DataFrame(rf.predict_proba(X_test),columns=['B','G'])
preds = prob['G']
 
#GBDT
from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(
  loss='ls'
, learning_rate=0.1
, n_estimators=100
, subsample=1
, min_samples_split=30
, min_samples_leaf=60
, max_depth=4
, init=None
, random_state=None
, max_features=None
, alpha=0.9
, verbose=0
, max_leaf_nodes=None
, warm_start=False
)
gbdt.fit(X_train,y_train)
preds = gbdt.predict(X_test)

#GBDT
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(max_depth=2,max_features=12,n_estimators=100,loss = 'exponential'
,random_state = 936 , learning_rate = 0.1 ,min_samples_leaf = 50)
gbdt.fit(X_train,y_train)
pred = gbdt.predict(X_test)
pd.crosstab(y_test,preds)
print(gbdt.predict_proba(X_test))
prob = pd.DataFrame(gbdt.predict_proba(X_test),columns=['B','G'])
preds = prob['G']

#建立MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes = (15,12), activation = 'relu', max_iter = 100)
mlp.fit(X_train , y_train)
prob =pd.DataFrame(mlp.predict_proba(X_test),columns=['B','G'])
preds = prob['G']

#模型评估
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, preds)
GINI = (auc-0.5)*2
GINI

from sklearn.metrics import classification_report
print(classification_report(y_test, pred, digits=4))

#平均召回率
lr.score(X_test,y_test)
svc.score(X_test,y_test)
gbdt.score(X_test,y_test)
rf.score(X_test,y_test)

