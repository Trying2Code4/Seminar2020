import os
import pandas as pd
import numpy as np

#%% LOAD DATA

os.chdir(r"C:\Users\sanne\Documents\Master QM\Block 3\Seminar Case Studies\Data")

observations = pd.read_csv('Observations_Report.csv', sep=';')
game = pd.read_csv('Observations_Game.csv', sep=';')
offers = pd.read_excel('OfferDetails_neat.xlsx')

#%% SOME EXPLORATORY ANALYSIS

# Sparsity of data
1 - observations.shape[0]/(observations['USERID'].nunique()*observations['OFFERID'].nunique()) # 0.9492401665334129
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