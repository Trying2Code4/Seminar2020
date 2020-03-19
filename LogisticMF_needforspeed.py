#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:07:31 2020

@author: matevaradi
"""

##NOTES:
# - Make sure numba is imported ($ conda install numba )
# - Enter your path at line 23
# - you can set different proportions to the data splitting in line 242
# - the equivalent of onlyVars is excludeNonclickers in line 258
# - you can adjust the hyperparameters at lines 267-268

## PLELIMINARIES
import time
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
from numba import jit
import os
### ENTER YOUR PATH HERE:  ###
os.chdir("____________________________________")
from sklearn.model_selection import train_test_split
from Tools import encoder, test_predictions, train_test, get_test_pred



#%% LOGISTIC MATRIX FACTORIZATION

class LogisticMF():

    def __init__(self, clicks, received,num_factors, reg_param=0.6, lrate=1.0,
                 maxiter=100,stochastic=False,epsilon=0.00001):
        self.clicks = clicks            #click data in sparse format
        self.received = received
        self.num_users = clicks.shape[0]
        self.num_items = clicks.shape[1]
        self.num_factors = num_factors
        self.iterations = maxiter
        self.epsilon = epsilon
        self.reg_param = reg_param       #lambda
        self.lrate = lrate               #learning rate
        self.stochastic= stochastic      #stochastic gradient descent or full
    
    @jit
    def train_model(self):

        self.ones = np.ones((self.num_users, self.num_items))
        self.user_vectors = np.random.normal(scale=1/self.reg_param,
                                             size=(self.num_users,self.num_factors))
        self.item_vectors = np.random.normal(scale=1/self.reg_param,
                                             size=(self.num_items,self.num_factors))
        #self.user_biases = np.random.normal(size=(self.num_users, 1))
        #self.item_biases = np.random.normal(size=(self.num_items, 1))
        self.user_biases = np.zeros((self.num_users, 1))
        self.item_biases = np.zeros((self.num_items, 1))


        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))
        
        if self.stochastic:
            user_bias_deriv_sum = np.ones((self.num_users, 1)) #
            item_bias_deriv_sum = np.ones((self.num_items, 1)) #
        else:
            user_bias_deriv_sum = np.zeros((self.num_users, 1)) 
            item_bias_deriv_sum = np.zeros((self.num_items, 1)) 
        
        self.time = np.zeros(self.iterations) 
        
        
        for i in range(self.iterations):
            t0 = time.time()
            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            if self.stochastic:
                user_vec_deriv, user_bias_deriv = self.stochastic_deriv(True)
            else:
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
            if self.stochastic:
                item_vec_deriv, item_bias_deriv = self.stochastic_deriv(False)
            else:
                item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.lrate / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.lrate / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += np.multiply(bias_step_size, item_bias_deriv)
            
            #self.convergence[i,0]=self.log_likelihood()
            #self.convergence[i,1]=np.linalg.norm(user_vec_deriv)
            #self.convergence[i,2]=np.linalg.norm(item_vec_deriv)
            t2 = time.time()
            self.time[i]=(t2-t0)
            print('iteration %i finished in %f seconds' % (i + 1, t2 - t0))
            #print('Log-likelihood: ',self.convergence[i,0])
            #print("User gradient norm: ",self.convergence[i,1])
            #print("Item gradient norm: ",self.convergence[i,2])
            #P = self.predict()
            #results=test_predictions2(P,key,trainset,testset,replacement=rep)
            #RMSE=testRMSE(results)
            #self.convergence[i,3]=RMSE 
            #print('RMSE: ',RMSE)
            
            #Stop if relative change to log likelihood is smaller than epsilon
            #if abs((self.convergence[(i-1),0]-self.convergence[i,0])/self.convergence[(i-1),0])<self.epsilon:
            #    self.iterations=i
            #    self.convergence=self.convergence[0:(i+1),:]
            #    break
                
            

    @jit
    def deriv(self, user):
        if user:
            vec_deriv = self.clicks.dot(self.item_vectors) #
            #bias_deriv = np.expand_dims(np.sum(self.clicks, axis=1), 1)
            bias_deriv = np.sum(self.clicks.multiply(self.received), axis=1) #

        else:
            vec_deriv = self.clicks.transpose().dot(self.user_vectors)    #
            #bias_deriv = np.expand_dims(np.sum(self.clicks, axis=0), 1)
            bias_deriv = np.sum(self.clicks.multiply(self.received), axis=0).T   #
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
    
    @jit
    def stochastic_deriv(self, user, batch=0.10):
        if user:
            sample=np.random.choice(self.num_items,size=int(batch*self.num_items))
            item_vector_sample=self.item_vectors[sample,:]  #dim si x f
            vec_deriv = self.clicks[:,sample] @ item_vector_sample #dim nu x f
            bias_deriv = np.sum(self.clicks[:,sample], axis=1) 
            
            A = np.dot(self.user_vectors, item_vector_sample.T) #dim nu x si
            A += self.user_biases
            A += self.item_biases[sample,:].T
            A = np.exp(A)
            A /= (A + self.ones[:,sample])
            A = self.received[:,sample].multiply(A)
            A = A.toarray()
            vec_deriv = vec_deriv - np.dot(A, item_vector_sample) 
            bias_deriv = bias_deriv - np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv = vec_deriv-  self.reg_param * self.user_vectors

        else:
            sample=np.random.choice(self.num_users,size=int(batch*self.num_users))
            user_vector_sample=self.user_vectors[sample,:]  #dim su x f
            vec_deriv = self.clicks[sample,:].transpose() @ user_vector_sample #dim ni x f
            bias_deriv = np.sum(self.clicks[sample,:], axis=0).T   
            
            A = np.dot(user_vector_sample, self.item_vectors.T) #dim su x ni
            A += self.user_biases[sample,:]
            A += self.item_biases.T
            A = np.exp(A)
            A /= (A + self.ones[sample,:])
            A = self.received[sample,:].multiply(A)
            A = A.toarray()
            vec_deriv = vec_deriv - np.dot(A.T, user_vector_sample)
            bias_deriv = bias_deriv - np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv = vec_deriv - self.reg_param * self.item_vectors
            
        return (vec_deriv, bias_deriv)
    
    @jit
    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = self.clicks.multiply(A) #
        B= self.received.multiply(B) #
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = self.received.multiply(A) #
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    @jit
    def predict(self):
        P = np.dot(self.user_vectors, self.item_vectors.T)
        P += self.user_biases
        P += self.item_biases.T
        P = np.exp(P)
        P = P/(self.ones+P)
        return P
    
    

   
#%% RUN

# Loading datasets
obs=pd.read_csv('df_obs.csv')

# Splitting to five different sizes:
proportions=np.array([0.2,0.4,0.6,0.8,1]) 
sizes=np.floor(proportions*obs.shape[0]) # The five different subset sizes
obsSorted=obs.sort_values(by=['USERID',"MailOffer"], axis = 0, ascending = True) # sorting the data
df0=obsSorted[0:int(sizes[0])]
df1=obsSorted[0:int(sizes[1])]
df2=obsSorted[0:int(sizes[2])]
df3=obsSorted[0:int(sizes[3])]
df4=obsSorted[0:int(sizes[4])]


# Array to save results:
iterTime=np.zeros(5)

i=0
for df in [df0,df1,df2,df3,df4]: 
    # Get data in correct format
    formatted,key=encoder(df,excludeNonclickers=True) 
    #Get data in sparse format and keys for mapping
    clicks=sp.csr_matrix((formatted["click"], (formatted["user"], formatted["item"])))
    #Load ones where there was an observation
    received=sp.csr_matrix((np.ones(len(formatted["user"])),(formatted["user"],formatted["item"])))

    # Train the model
    # Adjust hyperparameters
    iteration_num=11 # Number of iterations
    lambda_val=5     # Lambda
    factors=10       # Number of factors
    logMF=LogisticMF(clicks,received,num_factors=factors,maxiter=iteration_num,stochastic=True,reg_param=lambda_val) 
    logMF.train_model()
    iterTime[i]=np.mean(logMF.time[1:])
    i+1
