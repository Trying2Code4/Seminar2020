#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:07:31 2020

@author: matevaradi
"""

#Sparse matrix version

## PLELIMINARIES
import time
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import os
os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")

#Get a subset of the data from TrainTestSmall
from TrainTestSmall import trainset,testset
from Tools import encoder, test_predictions

#Get data in sparse format and keys for mapping
formatted,key=encoder(trainset) 

#%%  
## INTERMEZZO: how to deal with sparse matrixes

#Load our data to sprase format
sprs=sp.csr_matrix((formatted["click"], (formatted["user"], formatted["item"])))

#Load random vectors to try multiplications
R1=np.random.binomial(n=1, p=0.1, size=(2098, 10))
R2=np.random.binomial(n=1, p=0.1, size=(621, 2098))

#To return the dense version
sprs.toarray()  
sprs.toarray().shape

# Matrix multiplication
M= sprs @ R1
M= sprs.dot(R1)
# M will be a sparse matrix
M.shape
M.toarray()

# Element wise multiplication
M= sprs.multiply(R2)
# M will be a sparse matrix
M.toarray()
M.shape

#%%  
## LOAD DATA:

clicks=sp.csr_matrix((formatted["click"], (formatted["user"], formatted["item"])))
#Load ones where there was an observation
received=sp.csr_matrix((np.ones(len(formatted["user"])),(formatted["user"],formatted["item"])))

#%% LOGISTIC MATRIX FACTORIZATION

class LogisticMF():

    def __init__(self, clicks, received,num_factors, reg_param=0.6, lrate=1.0,
                 iterations=30):
        self.clicks = clicks            #click data in sparse format
        self.received = received
        self.num_users = clicks.shape[0]
        self.num_items = clicks.shape[1]
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param       #lambda
        self.lrate = lrate               #learning rate

    def train_model(self):

        self.ones = np.ones((self.num_users, self.num_items))
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        
        
        for i in range(self.iterations):
            t0 = time.time()
            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            user_vec_deriv, user_bias_deriv = self.deriv(True)
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            vec_step_size = self.lrate / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.lrate / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += np.multiply(bias_step_size, user_bias_deriv)

            #t1 = time.time()
            #print('iteration %i solved for users %f seconds' % (i + 1, t1 - t0))

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.lrate / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.lrate / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += np.multiply(bias_step_size, item_bias_deriv)
            t2 = time.time()

            print('iteration %i finished in %f seconds' % (i + 1, t2 - t0))

    def deriv(self, user):
        if user:
            vec_deriv = self.clicks.dot(self.item_vectors) #
            #bias_deriv = np.expand_dims(np.sum(self.clicks, axis=1), 1)
            bias_deriv = np.sum(self.clicks, axis=1)

        else:
            vec_deriv = self.clicks.transpose().dot(self.user_vectors)    #
            #bias_deriv = np.expand_dims(np.sum(self.clicks, axis=0), 1)
            bias_deriv = np.sum(self.clicks, axis=0).T
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.received.multiply(A)
        A = A.toarray()

        if user:
            vec_deriv = vec_deriv - np.dot(A, self.item_vectors)
            bias_deriv = bias_deriv - np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv = vec_deriv - self.reg_param * self.user_vectors
        else:
            vec_deriv = vec_deriv - np.dot(A.T, self.user_vectors)
            bias_deriv = bias_deriv - np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv = vec_deriv - self.reg_param * self.item_vectors
        return (vec_deriv, bias_deriv)
    
    # def stochastic_deriv(self, user, batch=0.10):
    #     if user:
    #         sample=np.random.choice(self.num_items,size=int(batch*self.num_items))
    #         item_vector_sample=self.item_vectors[sample,:]  #dim si x f
    #         vec_deriv = np.dot(self.clicks[:,sample], item_vector_sample) #dim nu x f
    #         bias_deriv = np.expand_dims(np.sum(self.clicks[:,sample], axis=1), 1)
            
    #         A = np.dot(self.user_vectors, item_vector_sample.T) #dim nu x si
    #         A += self.user_biases
    #         A += self.item_biases[sample,:].T
    #         A = np.exp(A)
    #         A /= (A + self.ones[:,sample])
    #         A = self.received[:,sample] * A
    #         vec_deriv -= np.dot(A, item_vector_sample) 
    #         bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)

    #         # L2 regularization
    #         vec_deriv -= self.reg_param * self.user_vectors

    #     else:
    #         sample=np.random.choice(self.num_users,size=int(batch*self.num_users))
    #         user_vector_sample=self.user_vectors[sample,:]  #dim su x f
    #         vec_deriv = np.dot(self.clicks[sample,:].T, user_vector_sample) #dim ni x f
    #         bias_deriv = np.expand_dims(np.sum(self.clicks[sample,:], axis=0), 1)

    #         A = np.dot(user_vector_sample, self.item_vectors.T) #dim su x ni
    #         A += self.user_biases[sample,:]
    #         A += self.item_biases.T
    #         A = np.exp(A)
    #         A /= (A + self.ones[sample,:])
    #         A = self.received[sample,:] * A
            
    #         vec_deriv -= np.dot(A.T, user_vector_sample)
    #         bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
    #         # L2 regularization
    #         vec_deriv -= self.reg_param * self.item_vectors
    #     return (vec_deriv, bias_deriv)
    
    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = self.clicks.multiply(A)   #
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    def predict(self):
        P = np.dot(self.user_vectors, self.item_vectors.T)
        P += self.user_biases
        P += self.item_biases.T
        P = np.exp(P)
        P = P/(self.ones+P)
        return P
    
#%% RUNNING THE METHOD
        
logMF=LogisticMF(clicks,received,num_factors=2,iterations=30)
logMF.train_model()

# Predictions
P=logMF.predict()
results=test_predictions(P,key,testset)


# Get Train RMSE

def trainRMSE(P,clicks,received):
    #Takes P matrix, clicks matrix and received matrix and outputs train RMSE
    P=logMF.predict()
    e=clicks-P
    e= np.multiply(e, received)
    e=np.square(e)
    RMSE=np.mean(e)
    RMSE=np.power(RMSE,0.5)
    return RMSE

trainRMSE(P,clicks,received)

# Get Test Predictions and Test RMSE

def testRMSE(results):
    #Takes results dataframe and outputs RMSE
    p=results["PROBABILITY"]
    click=results["CLICK"]
    e=click-p
    e*=e
    RMSE=np.mean(e)
    RMSE=np.power(RMSE,0.5)
    return RMSE

testRMSE(results)

#%% CROSS-VALIDATION

## Preparing parameters combinations

import itertools
# Parameters to try
fs=[2,3,4,5]                    # number of latent factors (f)
lambdas=[0.2,0.4,0.8,1,1.2]     # reg param (lamba)
deltas=[0.5,1,1,1.5]            # learning rate (delta)
# Calculate cartesian product of parameter values to try
param_combs=list(itertools.product(fs,lambdas,deltas))                                               
# Preparing to store probability matrices
num_combs=len(param_combs)


## Preparing Cross Validation

from Tools import CV_test_RMSE
from sklearn.model_selection import KFold
observations = pd.read_csv('Observations_Report.csv', sep=';')
nObs = 1000000
#Taking a subset of the observations
observationsSmall = observations.sample(frac=1)[1:nObs]
k=5   #number of folds to use
kf = KFold(n_splits=k)
kf.split(observationsSmall)
RMSEs=np.zeros((num_combs,k))  #parameter combinations in rows, folds in columns

## CV
# Loop through 5 folds
fold=0
P_matrices= []
for train_index, test_index in kf.split(observationsSmall):
    # Get data matrix
   train = observationsSmall.loc[observationsSmall.index[train_index].values]
   test =  observationsSmall.loc[observationsSmall.index[test_index].values]
   
   formatted,key=encoder(train) 
   clicks=sp.csr_matrix((formatted["click"], (formatted["user"], formatted["item"])))
   received=sp.csr_matrix((np.ones(len(formatted["user"])),(formatted["user"],formatted["item"])))

   # Train models on training set
   for comb in param_combs:
       logMF=LogisticMF(clicks,received,num_factors=comb[0],reg_param=comb[1], lrate=comb[2])
       logMF.train_model()
       P=logMF.predict()
       P_matrices.append(P)
   
   # Getting RMSEs on test set
   RMSEs[:,fold]=CV_test_RMSE(P_matrices, key, test)
   fold+=1
   P_matrices=[]

# Calculate average RMSEs over 5 folds
meanRMSE=np.mean(RMSEs,axis=1)

# Find the parameter combination with the lowest 
param_combs[np.argmin(meanRMSE)]
   
   
   
