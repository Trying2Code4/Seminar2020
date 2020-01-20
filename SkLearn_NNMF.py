import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from sklearn import metrics
from collections import defaultdict 
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#%% LOAD DATA

seed = 1

os.chdir(r"C:\Users\sanne\Documents\Master QM\Block 3\Seminar Case Studies\Data")

observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')

# Create row indices from user ids
temp = defaultdict(lambda: len(temp))
observations['ROW_IND'] = [temp[ele] for ele in observations['USERID']]
game['ROW_IND'] = [temp[ele] for ele in game['USERID']] # 1676 users not in observations
# Create column indices from offer ids
temp = defaultdict(lambda: len(temp))
observations['COL_IND'] = [temp[ele] for ele in observations['OFFERID']]
game['COL_IND'] = [temp[ele] for ele in game['OFFERID']] # 10 offers not in observations
del temp

# Create dataset without duplicates (keeping only the last observation of the duplicates)
unique_obs = observations.drop_duplicates(['USERID','OFFERID'], keep='last')
unique_obs = unique_obs[['ROW_IND','COL_IND','CLICK']]
sparse_all = csc_matrix((unique_obs['CLICK'], (unique_obs['ROW_IND'], unique_obs['COL_IND'])))
# =============================================================================
# MAYBE TODO: Remove duplicates while keeping only first observation to see if
# this makes any difference in results.
# =============================================================================

num_users, num_offers  = observations[['USERID','OFFERID']].nunique() # 297572, 2130

#%% NON-NEGATIVE MATRIX FACTORISATION using Scikit Learn (VERSION 1)

# Split sparse matrix (so both training and test set are sparse matrices)
trainset, testset = train_test_split(sparse_all, test_size = 0.2, random_state=seed)

# NMF on training set
nmf_v1 = NMF(n_components=10, init='nndsvd', random_state=seed)
result_v1 = nmf_v1.inverse_transform(nmf_v1.fit_transform(trainset)) # Filled-in matrix using MF
#predict_v1 = nmf_v1.inverse_transform(nmf_v1.transform(testset))
# =============================================================================
# TODO: Find out how to predict when testset has sparse (CSC) matrix format.
# Compute error measure.
# =============================================================================

#%% NON-NEGATIVE MATRIX FACTORISATION using Scikit Learn (VERSION 2)

# Split data, then declare training set as sparse matrix
trainset, testset = train_test_split(unique_obs, test_size = 0.2, random_state=seed)
sparse_v2 = csc_matrix((trainset['CLICK'], (trainset['ROW_IND'], trainset['COL_IND'])), shape=(num_users, num_offers))

# NMF on training set
nmf_v2 = NMF(n_components=10, init='nndsvd', random_state=seed)
result_v2 = nmf_v2.inverse_transform(nmf_v2.fit_transform(sparse_v2)) # Filled-in matrix using MF

# Prediction
testset.loc[:,'PREDICTION'] = result_v2[testset.loc[:,'ROW_IND'], testset.loc[:,'COL_IND']]

# Performance measures
nmf_v2_mse = metrics.mean_squared_error(testset['CLICK'], testset['PREDICTION']) # MSE 0.019893710454338503
nmf_v2_sse = nmf_v2_mse * testset.shape[0] # SSE 127652.84719561387l

#%% NON-NEGATIVE MATRIX FACTORISATION using Scikit Learn (VERSION 2 -- WITH CV & PARAM SELECTION)

# =============================================================================
# TODO: Switch order of for loops, so data is split just 5 times in total,
# instead of 5 times per param. Think about how to store the performance
# measures after switch.
# =============================================================================

#params = [2,5,10,20,50]
params = [10]
mean_mse_v2, mean_sse_v2 = [], []
cv = KFold(n_splits=5, shuffle=True, random_state = seed)

for c in params:
    nmf_v2cv = NMF(n_components=c, init='nndsvd', random_state=seed)
    mse_v2cv, sse_v2cv = [], []
    
    for train_v2cv, test_v2cv in cv.split(unique_obs):
        # Construct training set and convert into sparse matrix
        trainset_v2cv = unique_obs.iloc[train_v2cv]
        sparse_v2cv = csc_matrix((trainset_v2cv['CLICK'], (trainset_v2cv['ROW_IND'], trainset_v2cv['COL_IND'])), shape=(num_users, num_offers))
        
        # Apply NMF
        result_v2cv = nmf_v2cv.inverse_transform(nmf_v2cv.fit_transform(sparse_v2cv)) # Filled-in matrix using MF

        # Construct and predict test set
        testset_v2cv = unique_obs.iloc[test_v2cv]
        testset_v2cv.loc[:,'PREDICTION'] = result_v2cv[testset_v2cv.loc[:,'ROW_IND'], testset_v2cv.loc[:,'COL_IND']]
        
        # Performance measures
        error_sq = (testset_v2cv['CLICK']-testset_v2cv['PREDICTION'])**2
        mse_v2cv.append(np.mean(error_sq))
        sse_v2cv.append(np.sum(error_sq))
    
    mse_v2cv = np.array(mse_v2cv)
    mean_mse_v2.append(np.mean(mse_v2cv))
    
    sse_v2cv = np.array(sse_v2cv)
    mean_sse_v2.append(np.mean(sse_v2cv))

#plt.plot(params, mean_mse_v2, '-', label='MSE')
#plt.axhline(y=0.02192184495444662, color='r', linestyle='-', label='In-sample baseline')
#plt.xlabel('Number of latent factors')
#plt.show()
#
#plt.plot(params, mean_sse_v2, '-', label='SSE')
#plt.xlabel('Number of latent factors')
#plt.show()

#%% NON-NEGATIVE MATRIX FACTORISATION -- FITTING ON ENTIRE DATASET

# Fitting on entire set of observations
nmf = NMF(n_components=10, init='nndsvd', random_state=seed)
result = nmf.inverse_transform(nmf.fit_transform(sparse_all)) # Filled-in matrix using MF
#users = nmf.fit_transform(sparse) # [n_users x n_components]
#items = nmf.components_ # [n_components x n_items]

# Predictions for game dataset (removed users and offers that were not observed for now)
game2 = game[(game['ROW_IND']<result.shape[0]) & (game['COL_IND']<result.shape[1])]
game2['PREDICTION'] = result[game2.iloc[:,3], game2.iloc[:,4]]
