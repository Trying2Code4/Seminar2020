library(dplyr)
library(readr)
library(softImpute)
library(tidyverse)
library(tictoc)
library(RcppArmadillo)
library(Rcpp)
library(ggplot2)

# sourceCpp("/Users/colinhuliselan/Documents/Master/Seminar/Seminar2020_V2/R/gammaui.cpp")
#sourceCpp("~/Dropbox/Uni/Master_Econometrie/Blok_3/Seminar2020/R/gammaui.cpp")
sourceCpp("gammaui.cpp")
source("MajorizationFunctions.R")

# Preparing data -------------------------------------------------------------------------
# This is how you should import the data.
# The sequence here is important. We want to have a continuous sequence, starting at 1
# for the indices for user and order in our training set.

# Import train and game (test) set from whereever you store them
df_train <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/Observations_Report.csv",
                       ";", escape_double = FALSE, trim_ws = TRUE)
df_test <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/Observations_Game.csv",
                      ";", escape_double = FALSE, trim_ws = TRUE)

# df_train <- read_delim("~/Google Drive/Seminar 2020/Data/Observations_Report.csv",
#                        ";", escape_double = FALSE, trim_ws = TRUE)
# df_test <- read_delim("~/Google Drive/Seminar 2020/Data/Observations_Game.csv",
#                       ";", escape_double = FALSE, trim_ws = TRUE)


# Merge to create indices
df_test$CLICK <- NA
df <- rbind(df_train, df_test)

#Combine Mail and Offer id's
df[,"MailOffer"] <- df %>%
  unite("MailOffer", c("MAILID" ,"OFFERID"), sep = "_") %>%
  select("MailOffer")

# Order on click first such that NA are at bottom (no missings indices in training data)
df <- df[order(df$CLICK), ]
df <- df %>% 
  mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))
df <- df %>% 
  mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))

# Make it neat
df <- df[order(df$USERID_ind), c("USERID_ind", "OFFERID_ind", "CLICK")]

# Create ratios of CLICK per offer or user (== 1 or == 0 indicates no variation)
df <- df %>%
  group_by(USERID_ind) %>%
  mutate(ratioU = mean(CLICK, na.rm = TRUE)) %>%
  ungroup()

df <- df %>%
  group_by(OFFERID_ind) %>%
  mutate(ratioO = mean(CLICK, na.rm = TRUE)) %>%
  ungroup()

# Split
df_test <- df[is.na(df$CLICK), ]
df_train <- df[!(is.na(df$CLICK)), ]

# Save. Use the df_train.RDS file in CV
saveRDS(df_train, "/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
saveRDS(df_test, "/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_test")

# saveRDS(df_train, "~/Google Drive/Seminar 2020/Data/df_train")
# saveRDS(df_test, "~/Google Drive/Seminar 2020/Data/df_test")


# Train/test pred. -----------------------------------------------------------------
# Makes predictions for a train/test split for the FULL training set
# Also, includes columns/rows with only 0 or 1

# Use "Preparing data" first to get the df_train object
#df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- df[ ,c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")]

# Setting parameters
factors <- 2
lambda <- 1
iter <- 100
initType <- 4
onlyVar <- TRUE
llh <- TRUE
rmse <- TRUE
epsilon <- 0.01

set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK")]
df_test <- split$df_test[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "ratioU", "ratioO", "prediction")]
rm("split")

output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, rmse, 
                  epsilon)

baseline <- baselinePred(df_train, df_test)

# Visualization
hist(output$prediction$prediction)
plot(output$parameters$logllh)

# Train/test pred. SUB --------------------------------------------------------------
# Makes predictions for a train/test split for A SUBSET of the training set
# Also, includes columns/rows with only 0 or 1

# Use "Preparing data" first to get the df_train object
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- df[df$USERID_ind < 10000, c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")]

# Setting parameters
factors <- 4
lambda <- 1
iter <- 200
initType <- 2
onlyVar <- TRUE
llh <- TRUE
rmse <- TRUE
epsilon <- 0.01

set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK")]
df_test <- split$df_test[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "ratioU", "ratioO", "prediction")]
rm("split")

output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)

baseline <- baselinePred(df_train2, df_test2)


# Visualization
hist(output2$prediction$prediction)
# xdata <- c(1:output$parameters$run)

plot(output$parameters$objective[1:sum(!is.na(output$parameters$objective))],
     col="blue", type = "l", ylab="Objective Function", xlab="Iteration")
plot(output$parameters$deviance[1:sum(!is.na(output$parameters$deviance))],
     col="green", type = "l", ylab="Deviance", xlab="Iteration")
plot(output$parameters$rmse[1:sum(!is.na(output$parameters$rmse))],
     col="red", type = "l", ylab="RMSE", xlab="Iteration")

# Cross validation -----------------------------------------------------------------------
# Import train set
# Make sure the names are correct
# df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- readRDS("C:/Users/sanne/Documents/Master QM/Block 3/Seminar Case Studies/Seminar R-Code/df_train")
df <- df[df$USERID_ind < 10000, c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")]


# Input whichever hyperparameters you want to test
FACTORS <- c(50)
LAMBDA <- c(1,5,10,25,50,100,250,500,1000,2500,5000,10000)
INITTYPE <- c(2)
ONLYVAR <- c(TRUE, FALSE)
folds <- 5
iter <- 1000
epsilon <- 1e-08
warm <- TRUE

CVoutput <- crossValidate(df, FACTORS, LAMBDA, INITTYPE, ONLYVAR, folds, iter, 
                          epsilon, warm)

CVoutput_mean <- CVoutput %>% group_by(epsilon, Specification) %>% summarise_all(mean)

# Visualizing output
CVoutput$Specification <- as.factor(CVoutput$Specification)

p <- ggplot(CVoutput, aes(x=Specification, y=RMSE)) + 
  geom_boxplot()
p
  
CVoutput$RMSE

# Final predictions ----------------------------------------------------------------------
# If you want predictions for the final set

# Import test and train set
df_train <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df_test <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_test")

#Caclulcating parameters
#Hyperparameters
factors <- 2
priorsdu <- 1
priorsdi <- 1
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi

pars <- getPars(df_train[ ,c("USERID_ind", "OFFERID_ind", "CLICK")], 
                factors, priorsdu, priorsdi, priorlambdau, priorlambdai)

gameResults <- getPredict(df_test, pars$alpha, pars$beta, pars$C, pars$D)

# SOME TESTING ---------------------------------------------------------------------------

# See whether the indices are made correctly
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df_train <- trainTest(df)$df_train

max(df_train$USERID_ind_new)
length(unique(df_train$USERID_ind_new))

max(df_train$OFFERID_ind_new)
length(unique(df_train$OFFERID_ind_new))

# General parameter estimation algorithm testing
factors <- 2
priorsdu <- 2.5
priorsdi <- 2.5
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi
iter <- 0
initType <- 1


df_trainOrg <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df_testOrg <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_test")

df_train <- df_train[order(df_train$USERID_ind, df_train$OFFERID_ind), ]
df_trainOrg <- df_train[order(df_train$USERID_ind, df_train$OFFERID_ind), ]

sum(df_train$ratioU - df_trainOrg$ratioU)

sum(is.na(df_train$ratioU))
sum(is.na(df_trainOrg$ratioU))









alpha <- df %>%
  group_by(USERID_ind) %>%
  summarize(meanCLICK = mean(CLICK))

temp <- df %>%
  group_by(USERID_ind) %>%
  summarize(meanCLICK = mean(CLICK)) %>%
  select(meanCLICK)


alpha <- -1 * log(1/meanCLICK - 1)

beta <- rep(0, ni)


pars <- parEst(parEst(df_train, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, initType))

C <- output$parameters$C
D <- output$parameters$D
alpha <- output$parameters$alpha
beta <- output$parameters$beta

test <- getPredict(df_test[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "prediction")], 
                   alpha, beta, C, D)

test$prediction[is.na(test$prediction)] <- 0

hist(test$prediction)

# Problem with indices ------------------------------------------------------------------

df_test <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_test")
