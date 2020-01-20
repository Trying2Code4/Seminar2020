import os
import pandas as pd
import numpy as np

from surprise import SVDpp
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

#%% LOAD DATA

os.chdir(r"C:\Users\sanne\Documents\Master QM\Block 3\Seminar Case Studies\Data")

observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')

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