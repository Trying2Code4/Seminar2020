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
#from BaselinePredictions import baselines 
from Tools import train_test
import matplotlib.pyplot as plt
import seaborn as sns
import time


# RMSE function
def RMSE(pred,true):
    return (np.sqrt(np.mean(np.square(pred-true))))


## Loading data
offers=pd.read_csv("offers_clean.csv")


#%% INDIVIDUAL MODEL
## First let's create model for a single userid for inspection
# Excluding users that don't have a click yet  and that have less than 100 observations
byUser=trainset.groupby(["USERID"])
filteredData=byUser.filter(lambda x: (x["CLICK"].sum()>=5) and (x["CLICK"].count()>50) )
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

# Look at model coefficients for a single ID
id=44405733
pd.DataFrame(np.vstack((np.array(predictors),userFits[id].coef_[0])).T)



#%% CREATE PLOTS
# Look into coefficients
_,_,userFits,userids=ContentBased(trainset,testset,offers,method="Logit",minobs=50,minclicks=5,normalize=True)
coefs=np.zeros((len(userids),len(predictors)+1))

iter=0
for id in userids:
    coefs[iter,1:]=userFits[id].coef_
    coefs[iter,0]=userFits[id].intercept_
    iter+=1
# Storing it in a dataframe
coefdf=pd.DataFrame(coefs)


# Get the 10 predictors that are the least often 0
inds=np.argpartition(coefdf.apply(lambda x: x[abs(x)>0].count(), axis=0),-41)[-41:].values
# Get rid of price original as that one is fucked
inds=inds[inds!=3]

# Selecting the label names and non-zero coefficient values to plot
selectlabels=[(["INTERCEPT"] + predictors)[i] for i in sorted(inds)[::-1]]
selectcoefs=[coefs[abs(coefs[:,i])>0,i] for i in sorted(inds)[::-1]]

# Creating the plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(selectcoefs, labels=selectlabels,vert=False)

#%%% MAIN CODE
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

#%% TESTING

trainset,testset=train_test(nObs=1000000)

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


# "PARAMETER" OPTIMIZATION
# What's the best method/minobs/minclicks combo ? - Cross Validation
from sklearn.model_selection import KFold
import itertools
observations = pd.read_csv('Observations_Report.csv', sep=';')
## Preparing parameters combinations
methods=["Logit"]              
minobservation=[10,20,30,40,50,60,70]    
minclick=[1,2,3,5,10]
penalty=["l1","l2"]
# Calculate cartesian product of parameter values to try
param_combs=list(itertools.product(methods,minobservation,minclick))                                               
# Preparing to store probability matrices
num_combs=len(param_combs)


## Validation set approach:
#RMSEs=np.zeros(num_combs)
#for i,comb in enumerate(param_combs):
#     RMSEs[i]=ContentBased(trainset,testset,offers,method=comb[0],minobs=comb[1],minclicks=comb[2])[1]  
#     print("Combination ",i," is ready")

## Cross Validation
#Taking a subsample of the users
sampleOfUsers=np.random.choice(observations["USERID"].unique(),30000)
observationsSmall = observations[observations["USERID"].isin(sampleOfUsers) ]
k=3   #number of folds to use
kf = KFold(n_splits=k)
kf.split(observationsSmall)
RMSEs=np.zeros((num_combs,k))  #parameter combinations in rows, folds in columns

# Loop through 5 folds
fold=0
for train_index, test_index in kf.split(observationsSmall):
    # Get data matrix
   train = observationsSmall.loc[observationsSmall.index[train_index].values]
   test =  observationsSmall.loc[observationsSmall.index[test_index].values]
   

   # Train models on training set and get RMSE on test set
   for i,comb in enumerate(param_combs):
       RMSEs[i,fold]=ContentBased(train,test,offers,method=comb[0],minobs=comb[1],minclicks=comb[2],penalty=comb[3])[1]
   
   print("Fold %i out of %i finished" % (fold+1,k))
   fold+=1

# Calculate average RMSEs over 5 folds
meanRMSE=np.mean(RMSEs,axis=1)

# Find the parameter combination with the lowest 
param_combs[np.argmin(meanRMSE)] 
    
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



#RR: RES RUN on combination of train and val set, to be tested on the res set
RRprobs,RR_testRMSE,RRfits,RRuserids = ContentBased(trainval,res,offers,method="Logit",penalty="l2",minobs=10,minclicks=5,normalize=False,includeRMSE=True)
np.savetxt("CBF_RRprobs.csv", RRprobs, delimiter=",",header="CBF")


#GR: GAME RUN. Run on obs, predict game
GRprobs,GRfits,GRuserids = ContentBased(obs,game,offers,method="Logit",penalty="l2",minobs=10,minclicks=5,normalize=False,includeRMSE=False)
np.savetxt("CBF_GRprobs.csv", GRprobs, delimiter=",",header="CBF")


#VR: VAL RUN on train set, to be tested on the val set
VRprobs,VR_testRMSE,VRfits,VRuserids = ContentBased(train,val,offers,method="Logit",penalty="l2",minobs=10,minclicks=5,normalize=False,includeRMSE=True)
np.savetxt("CBF_VRprobs.csv", VRprobs, delimiter=",",header="CBF")



RR_testRMSE
baseline=baselines(trainval,res)
print(baseline)
VR_testRMSE
baseline=baselines(train,val)
print(baseline)



#PR: PLOT RUN, with normalization on combination of train and val set, to create coefficient plots
PRprobs,PRfits,PRuserids = ContentBased(trainval,res,offers,method="Logit",penalty="l2",minobs=10,minclicks=5,normalize=True,includeRMSE=False)

# Create Plots
coefs=np.zeros((len(PRuserids),len(predictors)+1))

iter=0
for id in PRuserids:
    coefs[iter,1:]=PRfits[id].coef_
    coefs[iter,0]=PRfits[id].intercept_
    iter+=1
# Storing it in a dataframe
coefdf=pd.DataFrame(coefs)

# Get the 10 most important predictors
#inds=np.argpartition(coefdf.apply(lambda x: x[abs(x)>0].count(), axis=0),-10)[-10:].values
inds=np.argpartition(coefdf.apply(lambda x: abs(x).sum(), axis=0),-10)[-10:].values

# Selecting the label names and non-zero coefficient values to plot
selectlabels=[(["INTERCEPT"] + predictors)[i] for i in sorted(inds)[::-1]]
prettyselectlabels=["Spain","Greece","Egypt","Rooms","Half width","Discount","Star Rating",
                    "Original price","Offer position","(Intercept)"]
selectcoefs=[coefs[abs(coefs[:,i])>0,i] for i in sorted(inds)[::-1]]
# Loading data into a dataframe
coefvalArray=np.array([])
VarArray=np.array([])
for j in range(10):
    coefvalArray=np.append(coefvalArray,selectcoefs[j])
    VarArray=np.append(VarArray,np.array(len(selectcoefs[j])*[prettyselectlabels[j]]))
plotdf=pd.DataFrame(index=np.array(range(len(VarArray))),
             data=np.column_stack((coefvalArray,VarArray)),columns=["Coef","Name"])

np.savetxt("plotdf.csv", plotdf, delimiter=",")

# Creating the plot
import seaborn as sns
fig = plt.figure(figsize=(6.3,3))
ax = fig.add_subplot(111)
ax.boxplot(selectcoefs, labels=prettyselectlabels,vert=False)
plt.savefig("coefplot.pdf",bbox_inches="tight",pad_inches=.5)

