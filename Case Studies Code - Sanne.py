import os
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
#from sklearn import metrics
from collections import defaultdict 
#import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

#from surprise import SVDpp
#from surprise import Dataset
#from surprise import Reader
#from surprise.model_selection import cross_validate

#%% LOAD DATA

os.chdir("C:\\Users\\sanne\\Documents\\Master QM\\Block 3\\Seminar Case Studies")

observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')
offers = pd.read_excel('OfferDetails_neat.xlsx')

# Merge observations with offers
#merged = pd.merge(observations, offers,  how='left', left_on=['MAILID','OFFERID'], right_on = ['MAILID','OFFERID'])
#merged.head()

# Create new variable that combines the mail id and offer id
#observations['MAILOFFER'] = observations['MAILID'].astype(str) + observations['OFFERID'].astype(str)

#%% CREATE SPARSE MATRIX

# Convert user ids into row indices
temp = defaultdict(lambda: len(temp)) 
observations['ROW_IND'] = [temp[ele] for ele in observations['USERID']]
game['ROW_IND'] = [temp[ele] for ele in game['USERID']] # 1676 users who do not appear in observations
# Convert offer ids into column indices
temp = defaultdict(lambda: len(temp)) 
observations['COL_IND'] = [temp[ele] for ele in observations['OFFERID']]
game['COL_IND'] = [temp[ele] for ele in game['OFFERID']] # 10 offers that did not occur in observations
del temp

# Declare sparse matrix
unique_obs = observations.drop_duplicates(['USERID','OFFERID'], keep='last')
sparse = csc_matrix((unique_obs['CLICK'], (unique_obs['ROW_IND'], unique_obs['COL_IND'])))

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
#%% BASELINE SSE

#metrics.mean_squared_error(observations['CLICK'], [0]*observations.shape[0]) # 0.02192184495444662
np.sum(observations['CLICK']**2) # 705292

#%% Non-negative Matrix Factorisation using Scikit Learn

nmf = NMF(n_components=10, init='nndsvd', random_state=1)
result = nmf.inverse_transform(nmf.fit_transform(sparse)) # Filled in matrix using MF
#users = nmf.fit_transform(sparse) # Matrix of shape (n_users x n_components) 
#items = nmf.components_ # Matrix of shape (n_components x n_items)


# Predictions for game dataset
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

#%% SVD++

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
