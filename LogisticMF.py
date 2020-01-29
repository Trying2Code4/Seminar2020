#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:13:51 2020

@author: matevaradi
"""

## PLELIMINARIES
import time
import numpy as np
import pandas as pd
import random
import os
os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")

#Get a subset of the data from TrainTestSmall
from TrainTestSmall import trainset,testset
from Tools import encoder, test_predictions

#Get data in sparse format and keys for mapping
formatted,key=encoder(trainset) 

#%%
## LOAD SPARSE REPRESENTATION TO n_u x n_i MATRIX Y


def load_matrix(sparsedata,weights=False):
    # Loading the (formatted) sparse data matrix of clicks
    t0 = time.time()
    num_users=sparsedata["user"].nunique()
    num_items=sparsedata["item"].nunique()
    
    clicks = np.zeros((num_users, num_items))
    received = np.zeros((num_users, num_items)) #w_ui matrix
    total = 0.0
    num_zeros = num_users * num_items
    
    for d in sparsedata.values:
        user = int(d[0])
        item = int(d[1])
        click = int(d[2])
        clicks[user][item] = click
        received[user][item] = 1      
        total += click
        num_zeros -= 1
    if weights:
        alpha = num_zeros / total
        print('alpha %.2f' % alpha)
        clicks *= alpha
    
    t1 = time.time()
    print ('Finished loading matrix in %f seconds' % (t1 - t0))
    return clicks, received

#%%
## LOGISTIC MATRIX FACTORIZATION
    

class LogisticMF():

    def __init__(self, clicks, received,num_factors, reg_param=0.6, lrate=1.0,
                 iterations=30):
        self.clicks = clicks
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
            #print('gradient descent done')
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            vec_step_size = self.lrate / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.lrate / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv
            
            #t1 = time.time()
            #print('iteration %i solved for users %f seconds' % (i + 1, t1 - t0))

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.deriv(False)
            #print('gradient descent done')
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.lrate / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.lrate / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t2 = time.time()

            print('iteration %i finished in %f seconds' % (i + 1, t2 - t0))

    def deriv(self, user):
        if user:
            vec_deriv = np.dot(self.clicks, self.item_vectors)
            bias_deriv = np.expand_dims(np.sum(self.clicks, axis=1), 1)

        else:
            vec_deriv = np.dot(self.clicks.T, self.user_vectors)
            bias_deriv = np.expand_dims(np.sum(self.clicks, axis=0), 1)
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.received * A

        if user:
            vec_deriv -= np.dot(A, self.item_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.user_vectors
        else:
            vec_deriv -= np.dot(A.T, self.user_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.item_vectors
        return (vec_deriv, bias_deriv)
    
    def stochastic_deriv(self, user, batch=0.05):
        if user:
            #batch+=0.05
            sample=np.random.choice(self.num_items,size=int(batch*self.num_items))
            item_vector_sample=self.item_vectors[sample,:]  #dim si x f
            vec_deriv = np.dot(self.clicks[:,sample], item_vector_sample) #dim nu x f
            bias_deriv = np.expand_dims(np.sum(self.clicks[:,sample], axis=1), 1)
            A = np.dot(self.user_vectors, item_vector_sample.T) #dim nu x si
            A += self.user_biases
            A += self.item_biases[sample,:].T
            A = np.exp(A)
            A /= (A + self.ones[:,sample])
            A = self.received[:,sample] * A
            vec_deriv -= np.dot(A, item_vector_sample) 
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.user_vectors

        else:
            self=logMF
            #batch=0.05
            sample=np.random.choice(self.num_users,size=int(batch*self.num_items))
            user_vector_sample=self.user_vectors[sample,:]  #dim su x f
            vec_deriv = np.dot(self.clicks[sample,:].T, user_vector_sample) #dim ni x f
            bias_deriv = np.expand_dims(np.sum(self.clicks[sample,:], axis=0), 1)
            A = np.dot(user_vector_sample, self.item_vectors.T) #dim su x ni
            A += self.user_biases[sample,:]
            A += self.item_biases.T
            A = np.exp(A)
            A /= (A + self.ones[sample,:])
            A = self.received[sample,:] * A
            vec_deriv -= np.dot(A.T, user_vector_sample)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.item_vectors
        return (vec_deriv, bias_deriv)
    
    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.clicks
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

#%%     
## Running the method 

clicks,received=load_matrix(formatted,False)          
logMF=LogisticMF(clicks,received,num_factors=2)
logMF.train_model()

# Predictions
P=logMF.predict()
results=test_predictions(P,key,testset)


# Get Train RMSE

def trainRMSE(P,clicks,received):
    #Takes P matrix, clicks matrix and received matrix and outputs train RMSE
    P=logMF.predict()
    e=clicks-P
    e*=received
    e*=e
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




    






