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
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
# df <- df[, c("USERID_ind", "OFFERID_ind", "CLICK")]

# If we want to use subset:
df <- df[df$USERID_ind < 10000,]
df <- df[, c("USERID", "MailOffer", "CLICK")]


mean(df$CLICK)
mean(df_test$CLICK)
# Setting parameters
factors <- 5
lambda <- 20
iter <- 10
initType <- 2
onlyVar <- T
llh <- FALSE
rmse <- TRUE
epsilon <- 0.001

set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_ind", "OFFERID_ind", "CLICK")]
df_test <- split$df_test[ ,c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO", "prediction", "USERID", "MailOffer")]
globalMean <- split$globalMean
rm("split")

set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon, globalMean = globalMean)

baseline <- baselinePred(df_test, globalMean=globalMean)
debug(baselinePred)


# Visualization
hist(output$prediction$prediction)
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
df <- readRDS("df_train.RDS")
df <- df[df$USERID_ind < 10000, c("USERID", "MailOffer", "CLICK")]

# Input whichever hyperparameters you want to test
# FACTORS <- c(1, 2, 5, 10, 20)
FACTORS <- c(15)
LAMBDA <- c(250)
INITTYPE <- c(2)
ONLYVAR <- c(TRUE, FALSE)
folds <- 5
iter <- 10
epsilon <- 1e-06
warm <- TRUE

CV <- crossValidate(df, FACTORS, LAMBDA, INITTYPE, ONLYVAR, folds, iter, 
                          epsilon, warm)

CVoutput_mean <- CVoutput %>% group_by(epsilon, Specification) %>% summarise_all(mean)

# Visualizing output
CVoutput$Specification <- as.factor(CVoutput$Specification)

p <- ggplot(CVoutput, aes(x=Specification, y=RMSE)) + 
  geom_boxplot()
p
  
CVoutput$RMSE


# Cross val FAST -------------------------------------------------------------------------
df <- readRDS("df_train.RDS")
df <- df[df$USERID_ind < 10000, c("USERID", "MailOffer", "CLICK")]

# Input whichever hyperparameters you want to test
FACTORS <- c(15, 20)
LAMBDA <- c(250)
INITTYPE <- c(2)
ONLYVAR <- c(TRUE, FALSE)
iter <- 10
epsilon <- 1e-06
warm <- TRUE

CVfast <- cvFast(df, FACTORS, LAMBDA, INITTYPE, ONLYVAR, iter, epsilon, warm)


# Running algorithm using best parameters from CV ----------------------------------------
df_train <- readRDS("df_train")
df_train <- df_train[, c("USERID", "MailOffer", "CLICK")]

df_val <- readRDS("df_val")
df_val <- df_val[, c("USERID", "MailOffer", "CLICK")]


# df_obs <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_obs.csv",
#                      ",", escape_double = FALSE, trim_ws = TRUE)
# saveRDS(df_obs, file="/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_obs.RDS")

df_obs <- readRDS("df_obs.RDS")
df_train <- df_obs[df_obs$res == 0, c("USERID", "MailOffer", "CLICK")]
df_res <- df_obs[df_obs$res == 1, c("USERID", "MailOffer", "CLICK")]

# Input whichever hyperparameters you want to test
factors <- 10
lambda <- 5
iter <- 20000
initType <- 1
llh = T
rmse = F
epsilon <- 1e-06
onlyVar = TRUE

prep <- prepData(df_train, df_val, onlyVar)

train <- prep$df_train
test <- prep$df_test
globalMean <- prep$globalMean

timeT <- system.time({
set.seed(0)
MMconvergenceT <- fullAlg(train, test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon, globalMean = globalMean)
})

onlyVar = FALSE
prep <- prepData(df_train, df_res, onlyVar)
train <- prep$df_train
test <- prep$df_test
globalMean <- prep$globalMean


timeF <- system.time({
  set.seed(0)
  MMconvergenceF <- fullAlg(train, test, factors, lambda, iter, initType, llh, 
                            rmse, epsilon, globalMean = globalMean)
})


baselineBest <- baselinePred(test, globalMean=globalMean)


readRDS("outputbestcvlambda10fact20")

write.csv(outputbestcvlambda10fact20$prediction, file = "Maj_pred_l10f20ret3.csv")

# Fitting on a set -----------------------------------------------------------------------
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_train.RDS")
# df <- df[, c("USERID_ind", "OFFERID_ind", "CLICK")]
df <- df[, c("USERID", "MailOffer", "CLICK")]


# Setting parameters
factors <- 10
lambda <- 10
iter <- 2000
initType <- 2
onlyVar <- FALSE
llh <- FALSE
rmse <- FALSE
epsilon <- 1e-06

# Make correct indices
df <- df %>% 
  mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))
df <- df %>% 
  mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))
df <- df[ ,c("USERID_ind", "OFFERID_ind", "CLICK")]

tic("Total time to fit")
check <- parEst(df, factors, lambda, iter, initType, llh, rmse)
toc()

# Final predictions ----------------------------------------------------------------------
# If you want predictions for the final set

df_obs <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_obs.csv",
                     ",", escape_double = FALSE, trim_ws = TRUE)
df_game <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_game.csv",
                     ",", escape_double = FALSE, trim_ws = TRUE)

df_obs <- df_obs[, c("USERID", "MailOffer", "CLICK")]
df_game <- df_game[, c("USERID", "MailOffer", "CLICK")]

saveRDS(df_game, "/Users/colinhuliselan/Documents/Master/Seminar/SharedData/df_game.RDS")

prep <- prepData(df_obs, df_game, onlyVar = FALSE)
df_obs <- prep$df_train
df_game <- prep$df_test
globalMean <- prep$globalMean

# Hyperparameters
factors <- 10
lambda <- 10
iter <- 10000
initType <- 2
epsilon <- 1e-06

# Estimate parameters
pars <- parEst(df_obs[, c("USERID_ind", "OFFERID_ind", "CLICK")], factors, lambda, iter, 
               initType, llh=FALSE, rmse=FALSE, epsilon=epsilon)

# Get predictions
gameResults <- getPredict(df_game[, c("USERID_ind", "OFFERID_ind", "CLICK",
                                      "ratioU", "ratioO", "prediction", "USERID", "MailOffer")],
                          pars$alpha, pars$beta, pars$C, pars$D)
gameResults$predictions[is.na(gameResults$predictions)] <- globalMean

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

# Preparing data for comparing gradients ----------------------------------------------------------------
df <- readRDS("Data/df_obs.RDS")
df <- df[, c("USERID", "MailOffer", "CLICK")]

unique_ids <- unique(df$USERID)

# Sample using original userid
df_obs10k <- df[df$USERID %in% sample(unique_ids, 10000, replace = FALSE), ]
df_obs50k <- df[df$USERID %in% sample(unique_ids, 50000, replace = FALSE), ]
df_obs100k <- df[df$USERID %in% sample(unique_ids, 1000, replace = FALSE), ]

addIndices <- function(df) {
  df <- df %>% 
    mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))
  df <- df %>% 
    mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))
  
  return(df)
}

dfList <- list(df_obs10k, df_obs50k, df_obs100k)
dfList <- lapply(dfList, addIndices)

write.csv(dfList[[1]], file = "df_obs10k.csv")
write.csv(dfList[[2]], file = "df_obs50k.csv")
write.csv(dfList[[3]], file = "df_obs100k.csv")

# df_obs10k <- addIndices(df_obs10k)
# df_obs50k <- addIndices(df_obs50k)
# df_obs100k <- addIndices(df_obs100k)

# write.csv(df_obs10k, file = "df_obs10k.csv")
# write.csv(df_obs50k, file = "df_obs50k.csv")
# write.csv(df_obs100k, file = "df_obs100k.csv")

# Add code for parEst here \\\\\\\\\
df_obs10k <- dfList[[1]][,c("USERID_ind", "OFFERID_ind", "CLICK")]
df_obs50k <- dfList[[2]][,c("USERID_ind", "OFFERID_ind", "CLICK")]
df_obs100k <- dfList[[3]][,c("USERID_ind", "OFFERID_ind", "CLICK")]

# Save initialization
df_obs10k <- read_delim("Data/DataGradient/df_obs10k.csv", ",", escape_double = FALSE, trim_ws = TRUE)
df_obs10k <- df_obs10k[,c("USERID_ind", "OFFERID_ind", "CLICK")]

init10k <- initChoose(df_obs10k, factors = 10, lambda = 5, initType = 2)
write.csv(init10k$alpha, "alpha10k.csv")
write.csv(init10k$beta, "beta10k.csv")

df_obs50k <- read_delim("Data/DataGradient/df_obs50k.csv", ",", escape_double = FALSE, trim_ws = TRUE)
df_obs50k <- df_obs50k[,c("USERID_ind", "OFFERID_ind", "CLICK")]

init50k <- initChoose(df_obs50k, factors = 10, lambda = 5, initType = 2)
write.csv(init50k$alpha, "alpha50k.csv")
write.csv(init50k$beta, "beta50k.csv")

df_obs100k <- read_delim("Data/DataGradient/df_obs100k.csv", ",", escape_double = FALSE, trim_ws = TRUE)
df_obs100k <- df_obs100k[,c("USERID_ind", "OFFERID_ind", "CLICK")]

# Convergence ----------------------------------------------------------------------------
df <- readRDS("df_obs.RDS")
df <- df[, c("USERID", "MailOffer", "CLICK")]

unique_ids <- unique(df$USERID)

# Sample using original userid
set.seed(123)
df_obs10k <- df[df$USERID %in% sample(unique_ids, 10000, replace = FALSE), ]
df_obs50k <- df[df$USERID %in% sample(unique_ids, round(10^4.5), replace = FALSE), ]
df_obs100k <- df[df$USERID %in% sample(unique_ids, 100000, replace = FALSE), ]

addIndices <- function(df) {
  df <- df %>% 
    mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))
  df <- df %>% 
    mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))
  return(df)
}

df_obs10k <- addIndices(df_obs10k)
df_obs50k <- addIndices(df_obs50k)
df_obs100k <- addIndices(df_obs100k)

write.csv(dfList[[1]], file = "df_obs10k.csv")
write.csv(dfList[[2]], file = "df_obs50k.csv")
write.csv(dfList[[3]], file = "df_obs100k.csv")

factors <- 10
lambda <- 5
iter <- 20000
initType <- 2
epsilon <- 1e-06
onlyVar = FALSE

time10k <- system.time({
  set.seed(123)
  pars10k <- parEst(df_obs10k[ ,c("USERID_ind", "OFFERID_ind", "CLICK")], factors, lambda, iter, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon)
})
saveRDS(pars10k, "pars10k.RDS")

time50k <- system.time({
  set.seed(123)
  pars50k <- parEst(df_obs50k[ ,c("USERID_ind", "OFFERID_ind", "CLICK")], factors, lambda, iter, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon)
})
saveRDS(pars50k, "pars50k.RDS")

time100k <- system.time({
  set.seed(123)
  pars100k <- parEst(df_obs100k[ ,c("USERID_ind", "OFFERID_ind", "CLICK")], factors, lambda, iter, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon)
})
save.RDS(pars100k, "pars100k.RDS")
>>>>>>> 37ec603cfcc324c1e2cf43d72608eb1d9847ec4d

init100k <- initChoose(df_obs100k, factors = 10, lambda = 5, initType = 2)
write.csv(init100k$alpha, "alpha100k.csv")
write.csv(init100k$beta, "beta100k.csv")
