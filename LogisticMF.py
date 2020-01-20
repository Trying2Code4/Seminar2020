#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:44:54 2020

@author: matevaradi
"""

## PLELIMINARIES
import os
os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")

import time
import numpy as np
import pandas as pd
import random


#%%
## LOAD DATA

data=pd.read_csv("Click_nodup.csv",sep=';',names=["user","item","count"],header=0)
#Get a subset of the data
sample=data[data["user"]<1000]
def reindex_observations(sample):
    #Returns the data sample with user and item indexes starting from 0
    from collections import defaultdict
    # Convert user ids into row indices
    temp = defaultdict(lambda: len(temp))
    sample.loc[:,"user"] = [temp[ele] for ele in sample['user']]
     # Convert item ids into row indices
    temp = defaultdict(lambda: len(temp))
    sample.loc[:,"item"] = [temp[ele] for ele in sample['item']]
    del temp
    return sample

sample=reindex_observations(observationsSmall)

#Use small subset from  TrainTestSmall.py (trainSet)
sample=trainset



def load_matrix(filename,weights=True):
    # Loading a (subset) of the data matrix
    t0 = time.time()
    #data=pd.read_csv(filename,sep=';',names=["user","item","count"])
    #data=data[data["count"]!=0]
    data=sample
    num_users=data["user"].nunique()
    num_items=data["item"].nunique()
    
    counts = np.zeros((num_users, num_items))
    received = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    
    for d in data.values:
        user = int(d[0])
        item = int(d[1])
        if d[2]==0:          
            count = -1
        else:
            count=1
        counts[user][item] = count
        received[user][item] = 1      
        total += count
        num_zeros -= 1
    if weights:
        alpha = num_zeros / total
        print('alpha %.2f' % alpha)
        counts *= alpha
    
    t1 = time.time()
    print ('Finished loading matrix in %f seconds' % (t1 - t0))
    return counts, received

#%%
## LOGISTIC MATRIX FACTORIZATION

class LogisticMF():

    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

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
            vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()

            print('iteration %i finished in %f seconds' % (i + 1, t1 - t0))

    def deriv(self, user):
        if user:
            vec_deriv = np.dot(self.counts, self.item_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=1), 1)

        else:
            vec_deriv = np.dot(self.counts.T, self.user_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=0), 1)
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = (self.counts + self.ones) * A

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

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.counts + self.ones) * A
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
## RUNNING THE METHOD        

counts,received=load_matrix("Click_nodup.csv",False)          
logMF=LogisticMF(counts,2)
logMF.train_model()
logMF.predict()

## COUNTIG RMSE

def get_RMSE(predicted,true,received):
    P=logMF.predict()
    e=true-P
    e*=received
    e*=e
    RMSE=np.mean(e)
    RMSE=np.power(RMSE,0.5)
    return RMSE    

true,received=load_matrix("Click_nodup.csv",False)
true[true==-1]=0
P=logMF.predict()
get_RMSE(P,true,received)

logMF.item_vectors
