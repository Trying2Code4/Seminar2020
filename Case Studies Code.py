import os
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from sklearn import metrics
from collections import defaultdict 
#import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

#from surprise import SVDpp
#from surprise import Dataset
#from surprise import Reader
#from surprise.model_selection import cross_validate

#%% LOAD DATA

seed = 1

os.chdir("C:\\Users\\sanne\\Documents\\Master QM\\Block 3\\Seminar Case Studies\\Data")

observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')
offers = pd.read_excel('OfferDetails_neat.xlsx')

# Merge observations with offers
#merged = pd.merge(observations, offers,  how='left', left_on=['MAILID','OFFERID'], right_on = ['MAILID','OFFERID'])
#merged.head()

# Create new variable that combines the mail id and offer id
#observations['MAILOFFER'] = observations['MAILID'].astype(str) + observations['OFFERID'].astype(str)

# Convert user ids into row indices
temp = defaultdict(lambda: len(temp)) 
observations['ROW_IND'] = [temp[ele] for ele in observations['USERID']]
game['ROW_IND'] = [temp[ele] for ele in game['USERID']] # 1676 users not in observations
# Convert offer ids into column indices
temp = defaultdict(lambda: len(temp)) 
observations['COL_IND'] = [temp[ele] for ele in observations['OFFERID']]
game['COL_IND'] = [temp[ele] for ele in game['OFFERID']] # 10 offers not in observations
del temp

# Create dataset without duplicates (keeping only the last observation of the duplicates)
unique_obs = observations.drop_duplicates(['USERID','OFFERID'], keep='last')
# =============================================================================
# MAYBE TODO: Remove duplicates while keeping only first observation to see if
# this makes any difference in result. (Obviously very low priority, to do only
# after we have the final model)
# =============================================================================

#%% SOME EXPLORATORY ANALYSIS

# Sparsity of data
1 - observations.shape[0]/(len(np.unique(observations['USERID']))*offers.shape[0]) # 0.9572316276567127
# Number of unique users, mails, and offers
observations[['USERID','MAILID','OFFERID']].nunique() # 297572, 693, 2130
# Number of emails opened
observations.groupby(['USERID', 'MAILID']).ngroups # 12031803

# Frequency table nr of clicks
sum_c = observations[['USERID','CLICK']].groupby(['USERID']).sum()
freq = pd.concat([sum_c['CLICK'].value_counts().rename('Frequency'), (sum_c['CLICK'].value_counts()/len(sum_c['CLICK'])).rename('Percentage')], axis=1).reset_index()
# Click rate per user
click_rate = observations[['USERID','CLICK']].groupby(['USERID']).mean()
#click_rate[click_rate['CLICK']>0].hist()

# Number of User/Offer duplicates
observations.shape[0] - observations.groupby(['USERID','OFFERID']).ngroups # 89306
# Number of User/Offer/Click duplicates
observations.shape[0] - observations.groupby(['USERID','OFFERID','CLICK']).ngroups # 87732
# Difference (Number of duplicates in which users changed their preference)
observations.groupby(['USERID','OFFERID','CLICK']).ngroups - observations.groupby(['USERID','OFFERID']).ngroups # 1574

# Users/offers in game dataset that do not appear in observations
new_users = list(set(game['USERID']) - set(observations['USERID'])) # 1676
new_offers = list(set(game['OFFERID']) - set(observations['OFFERID'])) # 108
#%% BASELINE

# Majority rule (predict zero for everyone)
metrics.mean_squared_error(observations['CLICK'], [0]*observations.shape[0]) # MSE 0.02192184495444662
np.sum(observations['CLICK']**2) # SSE 705292

# Use overall click rate for as prediction

# Use individual click rate as prediction 

#%% NON-NEGATIVE MATRIX FACTORISATION using Scikit Learn (VERSION 1)

# Declare sparse matrix, then split data (so both training and test set are sparse matrices)
sparse = csc_matrix((unique_obs['CLICK'], (unique_obs['ROW_IND'], unique_obs['COL_IND'])))
trainset, testset = train_test_split(sparse, test_size = 0.2, random_state=seed)

# NMF on training set
nmf_v1 = NMF(n_components=10, init='nndsvd', random_state=seed)
result_v1 = nmf_v1.inverse_transform(nmf_v1.fit_transform(trainset)) # Filled-in matrix using MF
predict_v1 = nmf_v1.transform(testset)
# =============================================================================
# TODO: Find a way to compare predictions to compare predictions to test set;
# calculate error measures
# =============================================================================

#%% NON-NEGATIVE MATRIX FACTORISATION using Scikit Learn (VERSION 2)

# Split data, then declare training set as sparse matrix
trainset, testset = train_test_split(unique_obs, test_size = 0.2, random_state=seed)
sparse = csc_matrix((trainset['CLICK'], (trainset['ROW_IND'], trainset['COL_IND'])))

# NMF on training set
nmf_v2 = NMF(n_components=10, init='nndsvd', random_state=seed)
result_v2 = nmf_v2.inverse_transform(nmf_v2.fit_transform(sparse)) # Filled-in matrix using MF

# Prediction
nusers, noffers = result_v2.shape
nrow = testset.shape[0]
predict_v2 = np.empty([nrow,1])

# I'm sure there's a more efficient way of doing this, open to suggestions...
for i in range(nrow):
    predict_v2[i] = result_v2[testset.iloc[i,4], testset.iloc[i,5]]
testset['PREDICTION'] = predict_v2

# Error
mse_v2 = metrics.mean_squared_error(testset['CLICK'], testset['PREDICTION']) # MSE 0.019893710454338503
sse_v2 = mse_v2 * nrow # SSE 127652.84719561387

# =============================================================================
# TODO: 5-fold cv + selection of optimal n_components
# =============================================================================

#%% NON-NEGATIVE MATRIX FACTORISATION -- FITTING ON ENTIRE DATASET

# Fitting on entire set of observations
nmf = NMF(n_components=10, init='nndsvd', random_state=seed)
result = nmf.inverse_transform(nmf.fit_transform(sparse)) # Filled-in matrix using MF
#users = nmf.fit_transform(sparse) # [n_users x n_components]
#items = nmf.components_ # [n_components x n_items]

# Predictions for game dataset (removed users and offers that were not observed for now)
nusers, noffers = result.shape # 297572, 2130
game2 = game[(game['ROW_IND']<nusers) & (game['COL_IND']<noffers)]
nrow = game2.shape[0] # 8041274
predictions = np.empty([nrow,1])

for i in range(nrow):
    row_ind = game2.iloc[i,3]
    col_ind = game2.iloc[i,4]
    if row_ind < nusers and col_ind < noffers:
        predictions[i] = result[row_ind, col_ind]
game2['PREDICTION'] = predictions

#%% SVD++ (PREDICTIONS ARE NOT WORKING PROPERLY)

# Make a smaller subset to improve running times
dfsmall = observations.sort_values(by=['USERID'], axis = 0, ascending = True)
dfsmall = dfsmall[:10000]
dfsmall[['USERID']].max() # 44385777

game_small = game[game['USERID'] < int(dfsmall[['USERID']].max())]
game_small['MAILOFFER'] = game_small['MAILID'].astype(str) + game_small['OFFERID'].astype(str)
game_small = game_small[['USERID', 'MAILOFFER']]
game_small['RATING'] = float("NaN")
game_list = game_small.values.tolist()

# Adjust data such that it can be used by 'Surprise'
reader = Reader(rating_scale=(1,0))
data = Dataset.load_from_df(dfsmall[['USERID','MAILOFFER','CLICK']], reader)
trainset = data.build_full_trainset()

mf = SVDpp(random_state=1)
# cross_validate(mf, data, measures=['MSE'], cv=2, verbose=True)
mf.fit(trainset)
predictions = mf.test(game_list)
predictions = pd.DataFrame(predictions)
predictions.head()
predictions['est'].value_counts()
