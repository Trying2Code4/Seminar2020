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
os.chdir("\\\\campus.eur.nl\\users\\students\\495556mv\\Documents\\Seminar2020")
#os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Loading datasets
game=pd.read_csv('df_game.csv')
obs=pd.read_csv('df_obs.csv')
trainval=obs[obs["res"]==0]
res=pd.read_csv('df_res.csv')
train=pd.read_csv('df_train.csv')
val=pd.read_csv('df_val.csv')

# |train| |val | |res|  |game|
# |  trainval  |                   (here)
# |       obs        |

# Loading individual predictions from the VAL RUNs 
CBF=pd.read_csv('CBF_VRprobs.csv')
LMF=pd.read_csv('LMF_VRprobs.csv')
MAJ=pd.read_csv('Maj_pred_l10f20ret3.csv')
val["CBF"]=CBF
val["LMF"]=LMF
val["MAJ"]=MAJ["prediction"]

byUser=train[['CLICK','USERID']].groupby(["USERID"])
obs_clicks=byUser.agg(["count","sum"]) #number of observations and clicks on the train set
obs_clicks.reset_index(level=0,inplace=True)

# Loading relevant information: number of clicks and observations per user
val=pd.merge(val,obs_clicks, how="left", on=['USERID'])
val=val.rename(columns={("CLICK", "count"):"num_obs",("CLICK", "sum"):"num_clicks"})
# 0 for users that don't appear in the train set
val.loc[np.isnan(val["num_obs"]),["num_obs"]]=0
val.loc[np.isnan(val["num_clicks"]),["num_clicks"]]=0

#%% CREATE HYBRID METHODS

## H1) Weighted average based on linear regression
val["H1"]=val["USERID"]*0 # initialize
userids=val["USERID"].unique()


y=val["CLICK"]
X=val[["CBF","MAJ"]]
reg = LinearRegression(fit_intercept=False).fit(X, y)
coefs=np.array([max(reg.coef_[0],0),max(reg.coef_[1],0)])
weightH1=coefs[0]/np.sum(coefs)
    
val["H1"]=weightH1*val["CBF"]+ \
        (1-weightH1)*val["MAJ"]


 #%%   
## H2) Weighted average per user based on the number of observations
tau_k=10 # minimum number of observations in the content based model
weightH2=val["num_obs"].apply(lambda x: min(np.log(x-tau_k)/5,1) if (x-tau_k)>0 else 0 )
val["H2"]=(val["CBF"]*weightH2 + (1-weightH2)*val["MAJ"])

#%% 
## H3) Switching based on the number of observations and number of clicks
ts_k=200  # threshold on the number of observation
ts_l=47  # threshold on the number of clicks
val["H3"]=val["CBF"]*( (val["num_obs"]>ts_k) & (val["num_clicks"]>ts_l) )+val["MAJ"]*(1-( (val["num_obs"]>ts_k) & (val["num_clicks"]>ts_l)))

# Finding the best combination of ts_k and ts_l
import itertools
## Preparing parameters combinations           
ts_ks=[50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]    
ts_ls=list(range(5,50))
# Calculate cartesian product of parameter values to try
param_combs=list(itertools.product(ts_ks,ts_ls))
num_combs=len(param_combs)
RMSEs=np.zeros((num_combs,3))

i=0
for comb in param_combs:
    ts_k=comb[0]
    ts_l=comb[1]
    RMSEs[i,0]=ts_k
    RMSEs[i,1]=ts_l
    p=val["CBF"]*( (val["num_obs"]>ts_k) & (val["num_clicks"]>ts_l) )+val["MAJ"]*(1-( (val["num_obs"]>ts_k) & (val["num_clicks"]>ts_l)))
    e=val["CLICK"]-p
    RMSEs[i,2]=np.power(np.mean(np.square(e)),0.5)
    i+=1
    

#%% EVALUATE HYBRIDS

#Inputs:
# testset: a dataframe of a testset, with a CLICK column and the prediction of a given method in the column
# hybrid_name: string that gives the column name corresponding to the method
#Output: RMSE
def testRMSE(testset,hybrid_name):
    e=testset["CLICK"]-testset[hybrid_name]
    RMSE=np.power(np.mean(np.square(e)),0.5)
    return RMSE


testRMSE(val,"LMF")
testRMSE(val,"CBF")
testRMSE(val,"MAJ")
testRMSE(val,"H1")
testRMSE(val,"H2")
testRMSE(val,"H3")









