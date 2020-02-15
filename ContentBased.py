#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:46:01 2020

@author: matevaradi
"""
#%%
## PRELIMINARIES
import os
#os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

offers=pd.read_csv("offers_clean.csv")
from TrainTestSmall import trainset,testset
from BaselinePredictions import baselines 
from Tools import train_test

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

## Fitting Logistic Regression
logit=LogisticRegression(penalty='l1', solver='liblinear')
fit=logit.fit(X_train,y_train)

# Diagnostics
RMSE(fit.predict_proba(X_train)[:,1],y_train.values) # train error
RMSE(fit.predict_proba(X_test)[:,1],y_test.values) # test error


## Fitting Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 24)
# Train the model on training data
rfFit=rf.fit(X_train, y_train)
RMSE(rfFit.predict_proba(X_train)[:,1],y_train.values) # train error
RMSE(rfFit.predict_proba(X_test)[:,1],y_test.values) # test error


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

#%%%
#' Putting it together in a function
def ContentBased(trainset,testset,offers,method="Logit",minobs=100,minclicks=1):
#' Returns click probabilities for the test set, test RMSE and model fits per user
#' INPUTS:
#' @param: trainset: A subset of click data to train the models on 
#' @param: trainset: A subset of click data to test the models on  
#' @param: offer: Data of offers
#' @param: method: "Logit" or "RF". Specifies model type.
#' @param: minobs: Minimum number of observations per user. A model will be created for 
#' all users with at least this many observaionts.
#' @param: minclicks: Minimum number of clicks per user. A model will be created for 
#' all users with at least this many clicks.

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    #supressing warnings
    pd.options.mode.chained_assignment = None  # default='warn'

    ## LOADING DATA
    # Excluding users that don't have a click yet  and that have less than a given number of observations OR
    # those that click on everything
    byUser=trainset.groupby(["USERID"])
    filteredData=byUser.filter(lambda x: (x["CLICK"].sum()>=1) and (x["CLICK"].count()>=minobs)\
                               and (x["CLICK"].sum()!=x["CLICK"].count() ) )
    # Users (id's) that we need to create a model for
    userids=filteredData["USERID"].unique()
    # Users that click on everything:
    clickalls= byUser.filter(lambda x: x["CLICK"].sum()==x["CLICK"].count() )["USERID"].unique()

    
    # Get train and test data by combining click data with data of offers
    testData=pd.merge(testset,offers, how="left", on=['OFFERID',"MAILID"])
    #trainData=pd.merge(filteredData,offers, how="left", on=['OFFERID',"MAILID"])
    
    # Initializing predicted probabilities:
    testData["PROBABILITIES"]=0 
    # Predict 1 for those that click on everything:
    for id in clickalls:
        if id in testData["USERID"].unique():
            testData.loc[testData["USERID"]==id,"PROBABILITIES"]=1
        
    
    # PREPARING MODELS
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
    
    userFits = dict.fromkeys(userids, []) 
    
    # TRAINING MODELS FOR EACH USER        
    for id in userFits.keys():
        userTrainData=trainset.loc[trainset["USERID"]==id,["OFFERID","CLICK"]]
        # Link offer data to this data
        userTrainData=pd.merge(userTrainData, offers, how="left", on='OFFERID')
        y_train=userTrainData["CLICK"]
        X_train=userTrainData[predictors]
        # Fitting the model
        if method=="Logit":
            logit=LogisticRegression(penalty='l1', solver='liblinear',max_iter=200)
            fit=logit.fit(X_train,y_train)
        elif method=="RF":
            rf = RandomForestClassifier(n_estimators = 100, random_state = 24)
            fit=rf.fit(X_train,y_train)        
        userFits[id]=fit
            
        
    # Getting predictions from the models    
    for id in userids:
        if id in testData["USERID"].unique(): #only predict for user that are in the test set
            testData.loc[testData["USERID"]==id,"PROBABILITIES"]=\
                userFits[id].predict_proba(testData.loc[(testData["USERID"]==id),predictors])[:,1]  
    # Test RMSE
    testRMSE=RMSE(testData["PROBABILITIES"],testData["CLICK"])
    
    #return testData["PROBABILITIES"],testRMSE,userFits
    return testRMSE

#%% TESTING

trainset,testset=train_test(nObs=10000000)


## FINDING THE BEST METHOD
# Logit VS RF
# Baseline
baseline=baselines(trainset,testset)
# Logit
logitRMSE=ContentBased(trainset,testset,offers,"Logit",minobs=100,minclicks=1)
# Logit improvement over best baseline: 1.99 %
100*(np.min(baseline)-logitRMSE)/np.min(baseline)

rfRMSE=ContentBased(trainset,testset,offers,"RF",minobs=100,minclicks=1)
# RF improvement over best baseline: 0.6 %
100*(np.min(baseline)-rfRMSE)/np.min(baseline)



## FURTHER TESTING
# What about RMSE on only the users that we have models for?
byUser=trainset.groupby(["USERID"])
filteredTrain=byUser.filter(lambda x: (x["CLICK"].sum()>=1) and (x["CLICK"].count()>=100) )
filteredTrain=trainset[trainset.index.isin(filteredTrain.index.values)]
userids=filteredTrain["USERID"].unique()
filteredTest=testset.loc[testset["USERID"].isin(userids)]

logitRMSE=ContentBased(filteredTrain,filteredTest,offers,"Logit")
baseline=baselines(filteredTrain,filteredTest)
(np.min(baseline)-logitRMSE)/np.min(baseline)
# Content Based even higher improvement over baseline:  (as expected)


## "PARAMETER" OPTIMIZATION
# What's the best method/minobs/minclicks combo ? - Cross Validation
from sklearn.model_selection import KFold
import itertools
observations = pd.read_csv('Observations_Report.csv', sep=';')


## Preparing parameters combinations
methods=["Logit","RF"]              
minobservation=[50,100,150,200,500,1000]    
minclick=[1,2,3,5,10]
# Calculate cartesian product of parameter values to try
param_combs=list(itertools.product(methods,minobservation,minclick))                                               
# Preparing to store probability matrices
num_combs=len(param_combs)


## Preparing Cross Validation
nObs = 10000000
#Taking a subset of the observations
observationsSmall = observations.sort_values(by=['USERID'], axis = 0, ascending = False)[5000:(5000+nObs)] #-> random users
k=3   #number of folds to use
kf = KFold(n_splits=k)
kf.split(observationsSmall)
RMSEs=np.zeros((num_combs,k))  #parameter combinations in rows, folds in columns

## CV
# Loop through 5 folds
fold=0
for train_index, test_index in kf.split(observationsSmall):
    # Get data matrix
   train = observationsSmall.loc[observationsSmall.index[train_index].values]
   test =  observationsSmall.loc[observationsSmall.index[test_index].values]
   

   # Train models on training set and get RMSE on test set
   for i,comb in enumerate(param_combs):
       RMSEs[i,fold]=ContentBased(train,test,offers,method=comb[0],minobs=comb[1],minclicks=comb[2])  
   
   print("Fold %i out of %i finished" % (fold+1,k))
   fold+=1

# Calculate average RMSEs over 5 folds
meanRMSE=np.mean(RMSEs,axis=1)

# Find the parameter combination with the lowest 
param_combs[np.argmin(meanRMSE)]
    
        
