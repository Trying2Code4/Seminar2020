import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
seed = 1

# %% LOAD DATA

## Place the data in the folder that contains your script
#observations2 = pd.read_csv('~/Documents/Master/Seminar/Code/Data/Observations_Report.csv', sep=';')
observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')
seed = 1

# %% OPTIONAL: MAKE A SMALLER SUBSET AND TRAIN + TEST

## Make a smaller subset to improve running times
nObs = 1000000
observationsSmall = observations.sort_values(by=['USERID'], axis = 0, ascending = True)[1:nObs]
trainset, testset = train_test_split(observationsSmall, test_size = 0.2, random_state=seed)