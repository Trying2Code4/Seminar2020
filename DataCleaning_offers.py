#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:37:49 2020

@author: matevaradi
"""

## PREP

import pandas as pd
import numpy as np
import string
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale
import re

offers=pd.read_excel("OfferDetails_neat.xlsx")
maildate=pd.read_csv("MailDate.csv",sep=";")

## CLEANING 

# Getting rid of nans and inconsistent data types
offers.loc[offers["REVIEW_RATING"]=="-","REVIEW_RATING"]=np.nan
offers["REVIEW_RATING"]=offers["REVIEW_RATING"].astype(float)

offers.loc[offers["DURATION"].apply( lambda x: type(x) is not int),"DURATION"]
offers=offers[offers.DURATION != 'Prijs p.p. (o.b.v. 2 personen)']
offers["DURATION"]=offers["DURATION"].astype(int)

offers["STAR_RATING"]=offers["STAR_RATING"].astype(float)

offers[offers["PRICE"]=="Vertrek: 6 mei"]=offers[offers["PRICE"] != "Vertrek: 6 mei"]["PRICE"].mean()
offers["PRICE"]=offers["PRICE"].astype(float)
offers["PRICE"]=offers["PRICE"].fillna(offers["PRICE"].mean())
offers["PRICE_ORIGINAL"]=offers["PRICE_ORIGINAL"].fillna(offers["PRICE_ORIGINAL"].mean())


offers["OFFER_POSITION"]=offers["OFFER_POSITION"].astype(int)

#Countries 
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x[0:4]=="Tune" ),"COUNTRY_NAME"]='Tunesie'
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x[0:4]=="Isra" ),"COUNTRY_NAME"]='Israel'
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x[0:4]=="Ital" ),"COUNTRY_NAME"]='Italie'
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x[0:4]=="Kroa" ),"COUNTRY_NAME"]='Kroatie'
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x[0:4]=="Kaap" ),"COUNTRY_NAME"]='Kaapverdie'
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x=="Egypte " ),"COUNTRY_NAME"]="Egypte"
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x=="Spanje " ),"COUNTRY_NAME"]="Spanje"
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x=="Costa del Sol" ),"COUNTRY_NAME"]="Spanje"
offers.loc[offers["COUNTRY_NAME"].apply( lambda x: x=="Roda" ),"COUNTRY_NAME"]="Griekenland"

offers=pd.get_dummies(offers, columns=['COUNTRY_NAME'])

# Discounts
offers["DISCOUNT"]=1-offers["PRICE"]/offers["PRICE_ORIGINAL"] #discount (between 0 and 1)

# Meal Plan

# Dummies first 
# convert everything to lowercase
offers["MEAL_PLAN"]=offers["MEAL_PLAN"].apply( lambda x: x.lower())
#removing duplicate phrasings of the same thing
offers.loc[offers["MEAL_PLAN"]=="logies met ontbijt","MEAL_PLAN"]="logies en ontbijt"
offers.loc[offers["MEAL_PLAN"]=="logies ontbijt","MEAL_PLAN"]="logies en ontbijt"
offers.loc[offers["MEAL_PLAN"]=="logies & ontbijt","MEAL_PLAN"]="logies en ontbijt"
offers.loc[offers["MEAL_PLAN"]=="ail inclusive","MEAL_PLAN"]="all inclusive"
offers.loc[offers["MEAL_PLAN"]=="all inlcusive","MEAL_PLAN"]="all inclusive"
offers.loc[offers["MEAL_PLAN"]=="half pension","MEAL_PLAN"]="halfpension"
offers.loc[offers["MEAL_PLAN"]=="logies ","MEAL_PLAN"]="logies"

offers=pd.get_dummies(offers, columns=['MEAL_PLAN'])

# Convert to ordinal scale
offers["MEAL_PLAN_ord"]=offers["MEAL_PLAN_logies en ontbijt"]
offers["MEAL_PLAN_ord"]=offers["MEAL_PLAN_ord"]+offers["MEAL_PLAN_halfpension"].apply(lambda x: 2 if x==1 else 0)
offers["MEAL_PLAN_ord"]=offers["MEAL_PLAN_ord"]+offers["MEAL_PLAN_volpension"].apply(lambda x: 3 if x==1 else 0)
offers["MEAL_PLAN_ord"]=offers["MEAL_PLAN_ord"]+offers["MEAL_PLAN_all inclusive"].apply(lambda x: 4 if x==1 else 0)
offers["MEAL_PLAN_ord"]=offers["MEAL_PLAN_ord"]+offers["MEAL_PLAN_ultra all inclusive"].apply(lambda x: 5 if x==1 else 0)

offers["HALFWIDTH"]=offers["OFFER_VISUALISATION"].apply(lambda x: 1 if x=="06 - Acco half width" else 0)

# imputing missing values from the review rating column
imputer = KNNImputer(n_neighbors=5)
X=offers[["REVIEW_RATING","PRICE_ORIGINAL","STAR_RATING","COUNTRY_NAME_Bulgarije",
         "COUNTRY_NAME_Cyprus","COUNTRY_NAME_Egypte","COUNTRY_NAME_Griekenland","COUNTRY_NAME_Italie",
         "COUNTRY_NAME_Kaapverdie","COUNTRY_NAME_Kroatie","COUNTRY_NAME_Marokko",
         "COUNTRY_NAME_Malta","COUNTRY_NAME_Montenegro","COUNTRY_NAME_Portugal","COUNTRY_NAME_Spanje",
         "COUNTRY_NAME_Tunesie"]]
X["REVIEW_RATING"]=X["REVIEW_RATING"].apply(lambda x: np.nan if x==0 else x)
offers["REVIEW_RATING"]=imputer.fit_transform(X)[:,0]

# Room occupancy
offers["CHILDREN"]=offers["ROOM_OCCUPANCY"].apply(lambda x: 1 if "kind" in x.lower() else 0)
offers["ROOMS"]=offers["ROOM_OCCUPANCY"].apply(lambda x: int(re.search(r'\d+', x).group()) ) # return the first number

# Departure date
offers["DEPARTURE_FEBRUARY"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("feb" in str(x).lower() ) else 0)
offers["DEPARTURE_MARCH"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("mar" in str(x).lower() ) else 0)
offers["DEPARTURE_APRIL"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("apr" in str(x).lower() ) else 0)
offers["DEPARTURE_MAY"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("may" in str(x).lower() ) else 0)
offers["DEPARTURE_JUNE"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("jun" in str(x).lower() ) else 0)
offers["DEPARTURE_JULY"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("jul" in str(x).lower() ) else 0)
offers["DEPARTURE_AUGUST"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("aug" in str(x).lower() ) else 0)
offers["DEPARTURE_SEPTEMBER"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("sep" in str(x).lower() ) else 0)
offers["DEPARTURE_OCTOBER"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("oct" in str(x).lower() ) else 0)
offers["DEPARTURE_NOVEMBER"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("nov" in str(x).lower() ) else 0)
offers["DEPARTURE_DECEMBER"]=offers["DEPARTURE_DATE"].apply(lambda x: 1 if ("dec" in str(x).lower() ) else 0)

# Mail time
# Getting rid of NaNs first
mailmonth=offers["MAILID"].apply(lambda x \
    :"" if len(str(maildate[maildate["MAILID"]==x]["MAIL_DATETIME"]).split("-"))<2  \
    else str(maildate[maildate["MAILID"]==x]["MAIL_DATETIME"]).split("-")[1] )
offers["MAILMONTH"]=mailmonth
# Creating mail month dummies
offers["MAIL_FEBRUARY"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="2" else 0)
offers["MAIL_MARCH"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="3" else 0)
offers["MAIL_APRIL"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="4" else 0)
offers["MAIL_MAY"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="5" else 0)
offers["MAIL_JUNE"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="6" else 0)
offers["MAIL_JULY"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="7" else 0)
offers["MAIL_AUGUST"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="8" else 0)
offers["MAIL_SEPTEMBER"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="9" else 0)
offers["MAIL_OCTOBER"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="10" else 0)
offers["MAIL_NOVEMBER"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="11" else 0)
offers["MAIL_DECEMBER"]=offers["MAILMONTH"].apply(lambda x: 1 if x=="12" else 0)

offers.to_csv("offers_clean.csv",index=False)




