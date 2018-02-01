# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:59:02 2018

@author: 56999
"""

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
#import auto_bin_woe

import_x = X_filled
import_y = y

X_train,X_test, y_train, y_test = train_test_split(import_x,import_y,test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced')
rfecv = RFECV(estimator=lr, step=1, cv=StratifiedKFold(2),scoring='accuracy')
rfecv = rfecv.fit(X_train,y_train)
pred = rfecv.predict(X_test)
preds = rfecv.predict_proba(X_test)
print("Optimal number of features : %d" % rfecv.n_features_)
X_woe_rfe_selected = import_x.iloc[:,rfecv.support_].copy()

import matplotlib.pyplot as plt
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#pred = rfecv.predict(X_test)
#preds=rfecv.predict_proba(X_test)
prob = pd.DataFrame(preds,columns=['B','G'])
preds = prob['G']

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, preds)
GINI = (auc-0.5)*2
print(GINI)