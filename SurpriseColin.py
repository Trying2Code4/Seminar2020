import pandas as pd
import numpy as np
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import metrics
seed = 1

# %% LOAD DATA

## Place the data in the folder that contains your script
#observations2 = pd.read_csv('~/Documents/Master/Seminar/Code/Data/Observations_Report.csv', sep=';')
observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')

# %% OPTIONAL: MAKE A SMALLER SUBSET AND TRAIN + TEST

## Make a smaller subset to improve running times
nObs = 100000
seed = 1
observationsSmall = observations.sort_values(by=['USERID'], axis = 0, ascending = True)[1:nObs]
trainset, testset = train_test_split(observationsSmall, test_size = 0.2, random_state=seed)


## If we want to make a smaller game set too
# game_small = game[game['USERID'] < int(dfsmall[['USERID']].max())]
# game_small['MAILOFFER'] = game_small['MAILID'].astype(str) + game_small['OFFERID'].astype(str)
# game_small = game_small[['USERID', 'MAILOFFER']]
# game_small['RATING'] = float("NaN")
# game_list = game_small.values.tolist()



# %% DATA PREPARATION 

## Using https://surprise.readthedocs.io/en/stable/getting_started.html#load-from-folds-example
reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(observations[['USERID', 'OFFERID', 'CLICK']], reader)
## Build the training set
dataTrain = Dataset.load_from_df(trainset[['USERID', 'OFFERID', 'CLICK']], reader)
dataTrain = dataTrain.build_full_trainset()
## Build the test set
dataTest = testset[['USERID', 'OFFERID']]
dataTest['CLICK'] = float("NaN")
dataTest = dataTest.values.tolist()

# %% RUNNING 

## 1. Select the algorithm
algo = SVDpp(random_state=1)
## 2. Fit
algo.fit(dataTrain)
## 3. Predict
predictions = algo.test(dataTest)
predictions = pd.DataFrame(predictions)

# %% Evaluation

## RMSE
RMSE = np.sqrt(metrics.mean_squared_error(testset['CLICK'], predictions['est']))

## Basline cases:
## Case 1: Majority rule (predict zero for everyone)
## 1. MSE: 0.02192184495444662
baseline1 = [0]*testset.shape[0]
RMSE1 = np.sqrt(metrics.mean_squared_error(testset['CLICK'], baseline1))

## Case 2: Use overall click rate for prediction
## 1. MSE 0.021441277668239916
baseline2 = [np.mean(trainset['CLICK'])]*testset.shape[0]
RMSE2 = np.sqrt(metrics.mean_squared_error(testset['CLICK'], baseline2))

## Case 3: Use click rate per person as prediction using
## https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
## 1. Find averages per user in training set
temp = trainset.groupby('USERID')['CLICK'].mean().reset_index()
## 2. Assign these averages to users in test set
testsetInd = testset[['USERID','OFFERID']]
baseline3 = testsetInd.merge(temp, how='left', on = 'USERID')['CLICK']
## 3. Add the average click rate (in the train set) to missing users
#print(baseline3.isnull().sum())
baseline3 = baseline3.fillna(np.mean(trainset['CLICK']))
## 4. Calculate RMSE
RMSE3 = np.sqrt(metrics.mean_squared_error(testset['CLICK'], baseline3))

## Case 4: Use click rate per offer as prediction using
## https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
## 1. Find averages per offer in training set
temp = trainset.groupby('OFFERID')['CLICK'].mean().reset_index()
## 2. Assign these averages to users in test set
testsetInd = testset[['USERID','OFFERID']]
baseline4 = testsetInd.merge(temp, how='left', on = 'OFFERID')['CLICK']
## 3. Add the average click rate (in the train set) to missing users
#print(baseline4.isnull().sum())
baseline4 = baseline4.fillna(np.mean(trainset['CLICK']))
## 4. Calculate RMSE
RMSE4 = np.sqrt(metrics.mean_squared_error(testset['CLICK'], baseline4))

print(RMSE)
print(RMSE1)
print(RMSE2)
print(RMSE3)
print(RMSE4)

# %% Using built-in CV

## Run 5-fold cross-validation and print results
# data = Dataset.load_from_df(observationsSmall[['USERID', 'OFFERID', 'CLICK']], reader)
# cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
