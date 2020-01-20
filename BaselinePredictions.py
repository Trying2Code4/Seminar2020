import pandas as pd
import numpy as np
from sklearn import metrics

# %% LOAD DATA

## Place the data in the folder that contains your script
#observations2 = pd.read_csv('~/Documents/Master/Seminar/Code/Data/Observations_Report.csv', sep=';')
observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')

# %% BASELINE

## Note: Here I use the full observations set to compare the baseline predictions
## with the "real" values. In your case you probably substitute the observations
## dataframe with a test set

## Case 1: Majority rule (predict zero for everyone)
## 1. MSE: 0.02192184495444662
baseline1 = [0]*observations.shape[0]
metrics.mean_squared_error(observations['CLICK'], baseline1)

## Case 2: Use overall click rate for prediction
## 1. MSE 0.021441277668239916
baseline2 = [np.mean(observations['CLICK'])]*observations.shape[0]
metrics.mean_squared_error(observations['CLICK'], baseline2)

## Case 3: Use click rate per person as prediction using
## https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
## 1. MSE
baseline3 = observations.groupby('USERID')['CLICK'].transform('mean')
metrics.mean_squared_error(observations['CLICK'], baseline3)

## Case 4: Use click rate per offer as prediction
## 1. MSE
baseline4 = observations.groupby('OFFERID')['CLICK'].transform('mean')
metrics.mean_squared_error(observations['CLICK'], baseline4)

## Possible additional baseline methods.
## Case 5: Use the average of (1) avg user click rate and (2) avg offer click rate
## for each user-offer pair
