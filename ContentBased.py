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
os.chdir("\\\\campus.eur.nl\\users\\students\\495556mv\\Documents\\Seminar2020")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from Tools import train_test
import time


# RMSE function
def RMSE(pred,true):
    return (np.sqrt(np.mean(np.square(pred-true))))


## Loading data
offers=pd.read_csv("offers_clean.csv")



#%%% MAIN METHOD
#' Putting it together in a function
def ContentBased(trainset,testset,offers,method="Logit",minobs=100,minclicks=1,penalty="l1",normalize=False,includeRMSE=True):
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
#' @param: penalty: Type of penalty to use for Logit. "l1" or "l2"
#' @param: normalize: Whether to convert continous variables to z-scores before regression
#' @param: includeRMSE: whether to calculate the test RMSE.


    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    #supressing warnings
    pd.options.mode.chained_assignment = None  # default='warn'

    ## LOADING DATA
    # Excluding users that don't have a click yet  and that have less than a given number of observations OR
    # those that click on everything
    byUser=trainset.groupby(["USERID"])
    filteredData=byUser.filter(lambda x: (x["CLICK"].sum()>=minclicks) and (x["CLICK"].count()>=minobs)\
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
    
    print("Preparation ready")
    t0=time.time()
    # TRAINING MODELS FOR EACH USER
    i=1        
    for id in userFits.keys():
        userTrainData=trainset.loc[trainset["USERID"]==id,["OFFERID","CLICK"]]
        # Link offer data to this data
        userTrainData=pd.merge(userTrainData, offers, how="left", on='OFFERID')
        y_train=userTrainData["CLICK"]
        X_train=userTrainData[predictors]
        if normalize:
            # Normalize continous variables
            cont_vars=["OFFER_POSITION","REVIEW_RATING","PRICE_ORIGINAL","STAR_RATING","DISCOUNT","ROOMS"]
            X_train=X_train.apply(lambda x: (x-x.mean())/x.std() if (x.name in cont_vars and x.std()>0) else x,axis=0)
        
        # Fitting the model
        if method=="Logit":
            logit=LogisticRegression(penalty='l1', solver='liblinear',max_iter=200)
            fit=logit.fit(X_train,y_train)
        elif method=="RF":
            rf = RandomForestClassifier(n_estimators = 100, random_state = 24)
            fit=rf.fit(X_train,y_train)        
        userFits[id]=fit
        t1=time.time()
        print("Training model ",i," out of ",len(userids),". Time elapsed: ",t1-t0)
        i+=1
            
    i=1   
    # Getting predictions from the models    
    for id in userids:
        if id in testData["USERID"].unique(): #only predict for user that are in the test set
            testData.loc[testData["USERID"]==id,"PROBABILITIES"]=\
                userFits[id].predict_proba(testData.loc[(testData["USERID"]==id),predictors])[:,1]
        t1=time.time()        
        print("Getting predictions for the ",i,"th user out of ",len(userids),". Time elapsed: ",t1-t0)
        i+=1
    
    if includeRMSE:
        # Calculate test RMSE
        testRMSE=RMSE(testData["PROBABILITIES"],testData["CLICK"])
        return testData["PROBABILITIES"],testRMSE,userFits,userids
    else:
        
        return testData["PROBABILITIES"],userFits,userids

    
#%% FINAL RUNS

# Loading datasets
offers=pd.read_csv("offers_clean.csv")
game=pd.read_csv('df_game.csv')
obs=pd.read_csv('df_obs.csv')
trainval=obs[obs["res"]==0]
res=pd.read_csv('df_res.csv')
train=pd.read_csv('df_train.csv')
val=pd.read_csv('df_val.csv')

# |train| |val | |res|  |game|
# |  trainval  |                   (here)
# |       obs        |


# Running the models on the train data
probs,testRMSE,fits,userids = ContentBased(train,val,offers,method="Logit",penalty="l2",minobs=10,
                                           minclicks=5,normalize=False,includeRMSE=True)




