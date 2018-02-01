# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:06:43 2018

@author: 56999
"""

import numpy as np
import pandas as pd
import os

org_file_name = 'hulu_install_final.csv'
dir_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(dir_path+'\\..\\..\\test dataset\\'+org_file_name,encoding = "GBK")
#replace missing numeric values with nan in numpy
df = df.replace(-9999,np.nan)

x_columns = [x for x in df.columns if x not in ['id_no','gbflag','id','customer_id','loan_channel_code','third_order_no','created_at','_id','mobile']]
X = df[x_columns]
y = df['gbflag']
total_num = len(y)
n_good = len(y.loc[y == 1])
n_bad = total_num - n_good

#for string columns, calculate the bin and woe value
def bin_woe_str(var,varname,y,n_good,n_bad):
    var_list = set(var[varname])
    bin_list = np.arange(0,len(var_list))
    binned = var.replace(var_list, bin_list)
    woe_list = []
    var_gbflag = pd.concat([binned,y],1)
    for binn in bin_list:
        #calculate woe
        gb_list = var_gbflag.loc[var_gbflag[varname]==binn]['gbflag']
        n_good_in_bin = len(gb_list.loc[gb_list == 1])
        n_bad_in_bin = len(gb_list.loc[gb_list == 0])
        p_good = n_good and n_good_in_bin / n_good or 0
        p_bad = n_bad and n_bad_in_bin / n_bad or 0
        woe = np.log((p_good and p_good or 0.000001)/(p_bad and p_bad or 0.000001))
        woe_list.append(woe)
    woed = binned.replace(bin_list, woe_list)
    return woed

#for char features in the data set, do this to bin and woe
for varname in X.columns:
#   in case if there is only one value
    if len(set(X[varname])) > 1:
        if X[varname].dtypes in ['object']:
            result = bin_woe_str(pd.DataFrame(X[varname]),varname,y,n_good,n_bad)
#            X.drop(varname, axis="columns", inplace=True)
            X[varname] = result
    else:
        X= X.drop(varname, axis="columns")
        
#split data by different data source
emay_var_list = []
for varname in df.columns:
    if 'dlq' in varname:
        emay_var_list.append(varname)
df_emay = df[emay_var_list]

td_var_list = []
for varname in df.columns:
    if 'td' in varname:
        td_var_list.append(varname)
df_td = df[td_var_list]

xy_var_list = []
for varname in df.columns:
    if 'xy' in varname:
        xy_var_list.append(varname)
df_xy = df[xy_var_list]

sn_var_list = ['last_appear_phone','other_names_cnt',
               'search_orgs_cnt','in_court_blacklist',
               'in_p2p_blacklist','idcard_in_blacklist',
               'phone_in_blacklist','in_bank_blacklist',
               'last_appear_phone_in_blacklist','offline_cash_loan_cnt',
               'online_cash_loan_cnt','online_installment_cnt',
               'payday_loan_cnt','credit_card_repayment_cnt',
               'offline_installment_cnt','others_cnt',
               'search_cnt_recent_180_days',
               'search_cnt_recent_7_days','org_cnt_recent_180_days',
               'search_cnt_recent_90_days','org_cnt_recent_60_days',
               'search_cnt_recent_30_days',
               'search_cnt_recent_14_days','org_cnt_recent_14_days',
               'org_cnt_recent_30_days','org_cnt_recent_7_days',
               'org_cnt_recent_90_days','org_cnt',
               'search_cnt_recent_60_days','search_cnt']
for varname in df.columns:
    if 'sn' in varname:
        sn_var_list.append(varname)
df_sn = df[sn_var_list]

##check for all external data source to make sure that someone has a real empty record
#df_ext_src = pd.concat([df_emay,df_td.drop('id_no',axis = 1),df_xy.drop('id_no',axis = 1),df_sn.drop('id_no',axis = 1)],1)
#df_ext_empty = df_ext_src.loc[df_ext_src.drop('id_no',axis=1).notnull().sum(axis=1)==0]

#create a new dataset with no missing values
X_no_missing = X.dropna(axis=1)

#K-means classifier
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=6, random_state=150).fit_predict(X_no_missing)

X_class0 = X.loc[y_pred==0]
X_class1 = X.loc[y_pred==1]
X_class2 = X.loc[y_pred==2]
X_class3 = X.loc[y_pred==3]
X_class4 = X.loc[y_pred==4]
X_class5 = X.loc[y_pred==5]
#X_class6 = X.loc[y_pred==6]
#X_class7 = X.loc[y_pred==7]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='most_frequent',axis=0)
X_class0_filled = pd.DataFrame(imputer.fit_transform(X_class0.values))
X_class1_filled = pd.DataFrame(imputer.fit_transform(X_class1.values))
X_class2_filled = pd.DataFrame(imputer.fit_transform(X_class2.values))
X_class3_filled = pd.DataFrame(imputer.fit_transform(X_class3.values))
X_class4_filled = pd.DataFrame(imputer.fit_transform(X_class4.values))
X_class5_filled = pd.DataFrame(imputer.fit_transform(X_class5.values))
#X_class6_filled = pd.DataFrame(imputer.fit_transform(X_class6.values))
#X_class7_filled = pd.DataFrame(imputer.fit_transform(X_class7.values))

X_filled = pd.concat([X_class0_filled,X_class1_filled,X_class2_filled,X_class3_filled,X_class4_filled,X_class5_filled],0)

#transform into scale [-1,1]
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler((-1,1))
X_scaled = min_max_scaler.fit_transform(X_filled.values)