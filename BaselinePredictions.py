import pandas as pd
import numpy as np
from sklearn import metrics

# %% LOAD DATA

## Place the data in the folder that contains your script
#observations2 = pd.read_csv('~/Documents/Master/Seminar/Code/Data/Observations_Report.csv', sep=';')
observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')

# %% GENERAL

## Note: Here I use the full observations set to compare the baseline predictions
## with the "real" values. In your case you probably substitute the observations
## dataframe with a test set

## Case 1: Majority rule (predict zero for everyone)
## 1. MSE: 0.0219218449544466
baseline1 = [0]*observations.shape[0]
RMSE1 = np.sqrt(metrics.mean_squared_error(observations['CLICK'], baseline1))

## Case 2: Use overall click rate for prediction
## 1. MSE 0.021441277668239916
baseline2 = [np.mean(observations['CLICK'])]*observations.shape[0]
RMSE2 = np.sqrt(metrics.mean_squared_error(observations['CLICK'], baseline2))

## Case 3: Use click rate per person as prediction using
## https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
## 1. MSE
baseline3 = observations.groupby('USERID')['CLICK'].transform('mean')
RMSE3 = np.sqrt(metrics.mean_squared_error(observations['CLICK'], baseline3))

## Case 4: Use click rate per offer as prediction
## 1. MSE
baseline4 = observations.groupby('OFFERID')['CLICK'].transform('mean')
RMSE4 = np.sqrt(metrics.mean_squared_error(observations['CLICK'], baseline4))

## Print the results

print(RMSE1)
print(RMSE2)
print(RMSE3)
print(RMSE4)

# %% HOW TO APPLY IN A TRAIN TEST SCENARIO
from TrainTestSmall import trainset,testset

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

print(RMSE1)
print(RMSE2)
print(RMSE3)
print(RMSE4)


## Possible additional baseline methods.
## Case 5: Use the average of (1) avg user click rate and (2) avg offer click rate
## for each user-offer pair


# %% FUNCTION VERSION

def baselines(trainset,testset):
    ## Case 1: Majority rule (predict zero for everyone)
    baseline1 = [0]*testset.shape[0]
    RMSE1 = np.sqrt(metrics.mean_squared_error(testset['CLICK'], baseline1))
    
    ## Case 2: Use overall click rate for prediction
    baseline2 = [np.mean(trainset['CLICK'])]*testset.shape[0]
    RMSE2 = np.sqrt(metrics.mean_squared_error(testset['CLICK'], baseline2))
    
    ## Case 3: Use click rate per person as prediction 
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
    
    ## Case 4: Use click rate per offer as prediction
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
    
    print("1: Zeroes, 2: Overall click rate, 3: Click rate per user, 4: Click rate per offer")
    return RMSE1,RMSE2,RMSE3,RMSE4
    
