#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:46:01 2020

@author: matevaradi
"""
#%%
## PRELIMINARIES
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import statsmodels.api as sm

offers=pd.read_csv("offers_clean.csv")
from TrainTestSmall import trainset,testset

#RMSE function
def RMSE(pred,true):
    return (np.sqrt(np.mean(np.square(pred-true))))

## LOADING DATA

# Excluding users that don't have a click yet  and that have less than 100 observations
byUser=trainset.groupby(["USERID"])
filteredData=byUser.filter(lambda x: (x["CLICK"].sum()>0) and (x["CLICK"].count()>99) )
filteredData=trainset[trainset.index.isin(filteredData.index.values)]
#users (id's) that we need to create a model for
userids=filteredData["USERID"].unique()


predictors=['OFFER_POSITION', 'REVIEW_RATING', 'PRICE_ORIGINAL', 'STAR_RATING',
       'DISCOUNT','HALFWIDTH', 'CHILDREN','ROOMS', 'MEAL_PLAN_ord',
       'COUNTRY_NAME_Bulgarije', 'COUNTRY_NAME_Cyprus', 'COUNTRY_NAME_Egypte',
       'COUNTRY_NAME_Griekenland', 'COUNTRY_NAME_Israel',
       'COUNTRY_NAME_Italie', 'COUNTRY_NAME_Kaapverdie',
       'COUNTRY_NAME_Kroatie', 'COUNTRY_NAME_Malta', 'COUNTRY_NAME_Marokko',
       'COUNTRY_NAME_Montenegro', 'COUNTRY_NAME_Portugal',
       'COUNTRY_NAME_Spanje', 'COUNTRY_NAME_Tunesie', 'COUNTRY_NAME_Turkije',
       'COUNTRY_NAME_Verenigde Arabische Emiraten',
       'DEPARTURE_FEBRUARY', 'DEPARTURE_MARCH', 'DEPARTURE_APRIL','DEPARTURE_MAY',
       'DEPARTURE_JUNE', 'DEPARTURE_JULY', 'DEPARTURE_AUGUST',
       'DEPARTURE_SEPTEMBER', 'DEPARTURE_OCTOBER', 'DEPARTURE_NOVEMBER',
       'DEPARTURE_DECEMBER',
       'MAIL_FEBRUARY', 'MAIL_MARCH','MAIL_APRIL', 'MAIL_MAY', 'MAIL_JUNE',
       'MAIL_JULY', 'MAIL_AUGUST','MAIL_SEPTEMBER', 'MAIL_OCTOBER', 
       'MAIL_NOVEMBER', 'MAIL_DECEMBER']

#%%
## First let's create model for a single userid for inspection

id=userids[13]
# Training data
userTrainData=trainset.loc[trainset["USERID"]==id,["OFFERID","CLICK"]]
# Link offer data to this data
userTrainData=pd.merge(userTrainData, offers, how="left", on='OFFERID')
# Test data
userTestData=testset.loc[testset["USERID"]==id,["OFFERID","CLICK"]]
# Link offer data to this data
userTestData=pd.merge(userTestData, offers, how="left", on='OFFERID')


y_train=userTrainData["CLICK"]
X_train=userTrainData[predictors]
y_test=userTestData["CLICK"]
X_test=userTestData[predictors]

# Fitting the model
logit=LogisticRegression(penalty='l1', solver='liblinear')

fit=logit.fit(X_train,y_train)

# Diagnostics
RMSE(fit.predict_proba(X_train)[:,1],y_train.values) # train error
RMSE(fit.predict_proba(X_test)[:,1],y_test.values) # test error


#%%
## LOGIT PER USER

# initialize dict to store fit objects
userFits = dict.fromkeys(userids, []) 

# Loop through users to train models
for id in userFits.keys():
    userTrainData=trainset.loc[trainset["USERID"]==id,["OFFERID","CLICK"]]
    # Link offer data to this data
    userTrainData=pd.merge(userTrainData, offers, how="left", on='OFFERID')
    y_train=userTrainData["CLICK"]
    X_train=userTrainData[predictors]
    # Fitting the model
    logit=LogisticRegression(penalty='l1', solver='liblinear')
    fit=logit.fit(X_train,y_train)
    userFits[id]=fit
    

# Get test data
testData=pd.merge(testset,offers, how="left", on=['OFFERID',"MAILID"])

# Initialize predictions
testset["PROBABILITIES"]=0
pd.options.mode.chained_assignment = None  # default='warn'
# Get predictions    
for id in userids:
    testset.loc[testset["USERID"]==id,"PROBABILITIES"]=\
        userFits[id].predict_proba(testData.loc[(testData["USERID"]==id),predictors])[:,1]

# Test RMSE
RMSE(testset["PROBABILITIES"],testset["CLICK"])



