#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:29:12 2020

@author: matevaradi
"""

# Preliminaries
import time
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import os
#os.chdir("\\\\campus.eur.nl\\users\\students\\495556mv\\Documents\\Seminar2020")
os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Loading datasets
game=pd.read_csv('df_game.csv')
obs=pd.read_csv('df_obs.csv')
trainval=obs[obs["res"]==0]
res=pd.read_csv('df_res.csv')

# |train| |val | |res|  |game|
# |  trainval  |                   (here)
# |       obs        |

# Loading individual predictions from the RES RUNs 
CBF=pd.read_csv('CBF_RRprobs.csv')
LMF=pd.read_csv('LMF_RRprobs.csv')
MAJ=pd.read_csv('majorizationRes.csv')
res["CBF"]=CBF
res["LMF"]=LMF
res["MAJ"]=MAJ["prediction"]

byUser=trainval[['CLICK','USERID']].groupby(["USERID"])
obs_clicks=byUser.agg(["count","sum"]) #number of observations and clicks on the train set
obs_clicks.reset_index(level=0,inplace=True)

# Loading relevant information: number of clicks and observations per user
res=pd.merge(res,obs_clicks, how="left", on=['USERID'])
res=res.rename(columns={("CLICK", "count"):"num_obs",("CLICK", "sum"):"num_clicks"})
# 0 for users that don't appear in the train set
res.loc[np.isnan(res["num_obs"]),["num_obs"]]=0
res.loc[np.isnan(res["num_clicks"]),["num_clicks"]]=0

#%% CREATE HYBRID METHODS

## H1) Weighted average per user based on linear regression
res["H1"]=res["USERID"]*0 # initialize

weightH1=0.408543866379738 #result from Hybrid.py
    
res["H1"]=weightH1*res["CBF"]+ \
        (1-weightH1)*res["MAJ"]


 #%%   
## H2) Weighted average per user based on the number of observations
#tau_k=10 # minimum number of observations in the content based model
#weightH2=res["num_obs"].apply(lambda x: min(np.log(x-tau_k)/5,1) if (x-tau_k)>0 else 0 )
#res["H2"]=(res["CBF"]*weightH2 + (1-weightH2)*res["MAJ"])

#%% 
## H2) Switching based on the number of observations and number of clicks
ts_k=100  # threshold on the number of observation
ts_l=47  # threshold on the number of clicks
res["H2"]=res["CBF"]*( (res["num_obs"]>ts_k) & (res["num_clicks"]>ts_l) )+res["MAJ"]*(1-( (res["num_obs"]>ts_k) & (res["num_clicks"]>ts_l)))

#%% EVALUATE HYBRIDS

# Inputs:
# testset: a dataframe of a testset, with a CLICK column and the prediction of a given method in the column
# hybrid_name: string that gives the column name corresponding to the method
def testRMSE(testset,hybrid_name):
    e=testset["CLICK"]-testset[hybrid_name]
    RMSE=np.power(np.mean(np.square(e)),0.5)
    return RMSE


testRMSE(res,"LMF")
testRMSE(res,"CBF")
testRMSE(res,"MAJ")
testRMSE(res,"H1")
testRMSE(res,"H2")

#compare to baselines
baselines(trainval,res)



#%% ROC and PRECREC curves
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.metrics import average_precision_score

# ROC curve
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
y_test=res["CLICK"]

plt.figure(figsize=(6.3,5))
fpr, tpr, threshold = metrics.roc_curve(y_test, res["LMF"])
#roc_auc = metrics.auc(fpr, tpr)
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot(fpr, tpr, 'b', label = 'ISGA',color="gray",ls="solid")
fpr, tpr, threshold = metrics.roc_curve(y_test, res["MAJ"])
plt.plot(fpr, tpr, 'b', ls='--',lw=1.3, label='MAJ',color="gray")
fpr, tpr, threshold = metrics.roc_curve(y_test, res["CBF"])
plt.plot(fpr, tpr, 'b', ls='dotted',lw=1.3, label='CBF',color="gray")
fpr, tpr, threshold = metrics.roc_curve(y_test, res["H1"])
plt.plot(fpr, tpr, 'b', ls='solid',lw=1.3, label='H1',color="black")
fpr, tpr, threshold = metrics.roc_curve(y_test, res["H2"])
plt.plot(fpr, tpr, 'b', ls='dotted',lw=1.3, label='H2',color="black")

plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--',color="black")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate',fontsize=13)
plt.xlabel('False Positive Rate',fontsize=13)
plt.savefig("ROCcurve.pdf")
plt.show()


# PREC-RECALL curve
plt.figure(figsize=(6.3,5))
lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["LMF"])
plt.plot(lr_recall, lr_precision, ls='solid',lw=1.3, label='ISGA',color="gray")
lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["MAJ"])
plt.plot(lr_recall, lr_precision, ls='--',lw=1.3, label='MAJ',color="gray")
lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["CBF"])
plt.plot(lr_recall, lr_precision, ls='dotted',lw=1.3, label='CBF',color="gray")
lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["H1"])
plt.plot(lr_recall, lr_precision, ls='solid',lw=1.3, label='H1',color="black")
lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["H2"])
plt.plot(lr_recall, lr_precision, ls='dotted',lw=1.3, label='H2',color="black")

# axis labels
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
# show the legend
plt.legend()
# show the plot
plt.savefig("PRECcurve.pdf")
plt.show()


# AUC for PREC-RECALL curve
# ISGA
lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["LMF"])
lr_auc = metrics.auc(lr_recall, lr_precision)

lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["MAJ"])
lr_auc = metrics.auc(lr_recall, lr_precision)

lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["CBF"])
lr_auc = metrics.auc(lr_recall, lr_precision)

lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["H1"])
lr_auc = metrics.auc(lr_recall, lr_precision)

lr_precision, lr_recall, _ = precision_recall_curve(y_test, res["H2"])
lr_auc = metrics.auc(lr_recall, lr_precision)



