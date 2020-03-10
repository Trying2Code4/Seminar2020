library(softImpute)
library(tidyverse)
library(tictoc)
library(RcppArmadillo)
library(Rcpp)
library(openxlsx)

# library(dplyr)
# library(readr)
# library(ggplot2)

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


# Train/test pred. SUB --------------------------------------------------------------
# Makes predictions for a train/test split for A SUBSET of the training set
# Also, includes columns/rows with only 0 or 1

# Use "Preparing data" first to get the df_train object
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_train.RDS")
df <- df[, c("USERID_ind", "OFFERID_ind", "CLICK")]

# If we want to use subset:
df <- df[df$USERID_ind < 10000, ]

mean(df$CLICK)
mean(df_test$CLICK)
# Setting parameters
factors <- 3
lambda <- 5
iter <- 30
initType <- 2
onlyVar <- T
llh <- FALSE
rmse <- FALSE
epsilon <- 0.00000001

df_train <- df



check <- parEst(df_train, factors, lambda, iter, initType, llh, rmse)

set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK")]
df_test <- split$df_test[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "ratioU", "ratioO", "prediction")]
globalMean <- split$globalMean
rm("split")

set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon, globalMean = globalMean)

baseline <- baselinePred(df_test, globalMean=globalMean)
debug(baselinePred)


# Visualization
hist(output2$prediction$prediction)
# xdata <- c(1:output$parameters$run)

par(mfrow=c(2,2))
plot(output$parameters$objective[1:sum(!is.na(output$parameters$objective))],
     col="blue", type = "l", lwd=2, ylab="Objective Function", xlab="Iteration")
plot(output$parameters$deviance[1:sum(!is.na(output$parameters$deviance))],
     col="green", type = "l", lwd=2, ylab="Deviance", xlab="Iteration")
plot(output$parameters$rmse[1:sum(!is.na(output$parameters$rmse))],
     col="red", type = "l", lwd=2, ylab="RMSE", xlab="Iteration")
plot(output$parameters$factors[1:sum(!is.na(output$parameters$factors))],
     col="orange", type = "l", lwd=2, ylab="Number of factors", xlab="Iteration")
par(mfrow=c(1,1))

# Cross validation -----------------------------------------------------------------------
# Import train set
# Make sure the names are correct
# df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- readRDS("df_train")
df <- df[df$USERID_ind < 10000, c("USERID_ind", "OFFERID_ind", "CLICK")]

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

# Fitting on a set -----------------------------------------------------------------------
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_train.RDS")
df <- df[, c("USERID_ind", "OFFERID_ind", "CLICK")]

# Setting parameters
factors <- 3
lambda <- 5
iter <- 30
initType <- 2
onlyVar <- T
llh <- FALSE
rmse <- FALSE
epsilon <- 0.00000001

# Make correct indices
df <- df %>% 
  mutate(USERID_ind_new = group_indices(., factor(USERID_ind, levels = unique(USERID_ind))))
df <- df %>% 
  mutate(OFFERID_ind_new = group_indices(., factor(OFFERID_ind, levels = unique(OFFERID_ind))))
df <- df[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK")]

tic("Total time to fit")
check <- parEst(df, factors, lambda, iter, initType, llh, rmse)
toc()


# Final predictions ----------------------------------------------------------------------
# If you want predictions for the final set

# Import test and train set
df_train <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df_test <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_test")

#Caclulcating parameters
#Hyperparameters
factors <- 2
lambda <- 1

pars <- parEst(df_train[ ,c("USERID_ind", "OFFERID_ind", "CLICK")], factors, lambda, iter, initType)

gameResults <- getPredict(df_test, pars$alpha, pars$beta, pars$C, pars$D)

# Preparing data for mate ----------------------------------------------------------------
# This is how you should import the data.
# The sequence here is important. We want to have a continuous sequence, starting at 1
# for the indices for user and order in our training set.

# Import train and game set from whereever you store them
df_obs <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/Observations_Report.csv",
                       ";", escape_double = FALSE, trim_ws = TRUE)
df_game <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/Observations_Game.csv",
                      ";", escape_double = FALSE, trim_ws = TRUE)

# df_obs <- read_delim("~/Google Drive/Seminar 2020/Data/Observations_Report.csv",
#                        ";", escape_double = FALSE, trim_ws = TRUE)
# df_game <- read_delim("~/Google Drive/Seminar 2020/Data/Observations_Game.csv",
#                       ";", escape_double = FALSE, trim_ws = TRUE)

# Merge to create indices
df_game$CLICK <- NA
df <- rbind(df_obs, df_game)

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
# df <- df[order(df$USERID_ind), c("USERID_ind", "OFFERID_ind", "CLICK")]
df <- df[order(df$USERID_ind), ]

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
df_obs <- df[!(is.na(df$CLICK)), ]
df_game <- df[is.na(df$CLICK), ]

# Make a results set
df_obs$res <- rbinom(n = nrow(df_obs), size = 1, prob = 0.2)

# Make a validation set for combining the models
df_obs$val <- NA
df_obs$val[df_obs$res == 0] <-  rbinom(n = sum(df_obs$res == 0), size = 1, prob = 0.2)

# Splitting all the sets
df_res <- df_obs[df_obs$res == 1, ]
df_val <- df_obs[(df_obs$val == 1 & !is.na(df_obs$val)), ]
df_train <- df_obs[(df_obs$val == 0 & !is.na(df_obs$val)), ]

# Check if you want
nrow(df_res) + nrow(df_val) + nrow(df_train) - nrow(df_obs)

# Save everything
saveRDS(df_game, "/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_game")
write.csv(df_game, file = "df_game.csv")
saveRDS(df_obs, "/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_obs")
write.csv(df_obs, file = "df_obs.csv")
saveRDS(df_res, "/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_res")
write.csv(df_res, file = "df_res.csv")
saveRDS(df_val, "/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_val")
write.csv(df_val, file = "df_val.csv")
saveRDS(df_train, "/Users/colinhuliselan/Documents/Master/Seminar/Shared_Data/df_train")
write.csv(df_train, file = "df_train.csv")



