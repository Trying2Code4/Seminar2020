#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:29:12 2020

@author: matevaradi
"""

# Preliminaries
import numpy as np
import pandas as pd
import random
import os
#os.chdir("\\\\campus.eur.nl\\users\\students\\495556mv\\Documents\\Seminar2020")
os.chdir("/Users/matevaradi/Documents/ESE/Seminar/Seminar2020")
# Loading datasets
game=pd.read_csv('df_game.csv')
obs=pd.read_csv('df_obs.csv')
from sklearn.linear_model import LinearRegression



# |train| |val | |res|  |game|
# |  trainval  |                   (here)
# |       obs        |

# Loading individual predictions from the RES RUNs 
CBF=pd.read_csv('CBF_GRprobs.csv')
CBF=np.hstack([np.array([0]),CBF.values[:,0]])
MAJ=pd.read_csv("gamePred_MM.csv")
game["CBF"]=CBF
game["MAJ"]=MAJ["prediction"]


#%% CREATE HYBRID METHOD

## H1) Weighted average based on linear regression

weightH1=0.408543866379738 #result from Hybrid.py    
    
game["H1"]=weightH1*game["CBF"]+ \
        (1-weightH1)*game["MAJ"]

#%%
# SAVE RESULTS
game["PREDICTION"]=game["H1"]
game["USERID"]=game["USERID"].apply(int)
game[["USERID","MAILID","OFFERID","PREDICTION"]].to_csv("team2predictions.csv")
