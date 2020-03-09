library(dplyr)
library(readr)
library(softImpute)
library(tidyverse)
library(tictoc)
library(RcppArmadillo)
library(Rcpp)
library(ggplot2)
library(openxlsx)

# sourceCpp("/Users/colinhuliselan/Documents/Master/Seminar/Seminar2020_V2/R/gammaui.cpp")
#sourceCpp("~/Dropbox/Uni/Master_Econometrie/Blok_3/Seminar2020/R/gammaui.cpp")
sourceCpp("gammaui.cpp")
source("MajorizationFunctions.R")

### Simple simulation ----
### Trying a simulated dataset
n <- 10000
p <- 2000
f <- 3

# Underlying model
set.seed(123)
alpha <- runif(n, min=-5, max=0)
beta <- runif(p, min=-5, max=0)
C <- matrix(rnorm(n * f, 0, 2), n, f)
D <- matrix(rnorm(p * f, 0, 2), p, f)
gamma <- alpha + beta + C %*% t(D) + matrix(rnorm(n*p, 0, 1), n, p)
probability <- exp(gamma) / (1 + exp(gamma))
# hist(probability)

# Create a train subset with a certain sparsity level
sparsity <- 0.1
set.seed(123)
USERID <- sample(1:n, sparsity*n*p, replace = TRUE)
OFFERID <- sample(1:p, sparsity*n*p, replace = TRUE)
df <- data.frame("USERID" = USERID, "OFFERID" = OFFERID)
df <- unique(df)
df$CLICK <- probability[as.matrix(df[ , c("USERID", "OFFERID")])]
df$CLICK <- as.numeric(df$CLICK > 0.5)

# We can simply apply the trainTest and fullAlg now:

# Setting parameters
factors <- 3
lambda <- 20
iter <- 10
initType <- 2
onlyVar <- TRUE
llh <- TRUE
rmse <- TRUE
epsilon <- 0.0001

# Train test sploit
set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK")]
df_test <- split$df_test[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "ratioU", 
                             "ratioO", "prediction")]
rm("split")

# Getting predictions
set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)

# Plotting
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


### Complicated simulation ----
# We can also define our test set ourselves

# Create a test subset 20 percent of the train
testsize <- 0.2 * sparsity * n * p 

# We can use a different underlying process for the test set if we want
set.seed(123)
# alpha_t <- runif(n, min=-5, max=0)
alpha_t <- alpha
# beta <- runif(p, min=-5, max=0)
beta_t <- beta
# C <- matrix(rnorm(n * f, 0, 2), n, f)
C_t <- C
# D <- matrix(rnorm(p * f, 0, 2), p, f)
D_t <- D
gamma_t <- alpha_t + beta_t + C_t %*% t(D_t) + matrix(rnorm(n*p, 0, 1), n, p)
probability_t <- exp(gamma_t) / (1 + exp(gamma_t))

# Or make the entire test set 0 for example
# probability_t <- 0.1

# Make the dataframe
# Create a test subset with 20% of thr training size
testsize <- 0.2 * sparsity * n * p
set.seed(123)
USERID <- sample(1:n, testsize, replace = TRUE)
OFFERID <- sample(1:p, testsize, replace = TRUE)
df_test <- data.frame("USERID" = USERID, "OFFERID" = OFFERID)
df_test <- unique(df_test)
df_test$CLICK <- probability[as.matrix(df_test[ , c("USERID", "OFFERID")])]
df_test$CLICK <- as.numeric(df_test$CLICK > 0.5)

# Now predefine the test and train sets
df_train <- df
df_test$train_test <- 1
df_train$train_test <- 0
df <- rbind(df_train, df_test)

# Use the train test function
# User/offer click rate in train set (if user or offer not in train set, average is set at NaN)
df <- df %>% group_by(USERID) %>% mutate(ratioU = sum((!as.logical(train_test))*CLICK)/sum(!as.logical(train_test))) %>% ungroup()
df <- df %>% group_by(OFFERID) %>% mutate(ratioO = sum((!as.logical(train_test))*CLICK)/sum(!as.logical(train_test))) %>% ungroup()

df$prediction <- rep(NA, nrow(df))

# Set predictions of observations that have user mean zero AND are in the test set to zero
df$prediction[df$ratioU == 0 & as.logical(df$train_test)] <- 0

# Predict zeroes for users with mean zero, and remove them from the training set
if (onlyVar) {
  # Exclude observations that have user mean zero AND are in the training set
  df <- df[!(df$ratioU == 0 & !as.logical(df$train_test)), ]
}

# Create new indices. Make sure test is at bottom
df <- df[order(df$train_test), ]
df <- df %>% 
  mutate(USERID_ind_new = group_indices(., factor(USERID, levels = unique(USERID))))
df <- df %>% 
  mutate(OFFERID_ind_new = group_indices(., factor(OFFERID, levels = unique(OFFERID))))

# Split sets
df_test <- df[as.logical(df$train_test), ]
df_train <- df[!(as.logical(df$train_test)), c("USERID_ind_new", "OFFERID_ind_new", "CLICK", 
                                               "ratioU", "ratioO")]

# Getting predictions
set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)

### Overwrite simulation ----
# We can also simply overwrite the test clicks
df_test$CLICK <- sample(c(0,1), nrow(df_test), replace = TRUE)

# Getting predictions
set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)

mean(output$prediction$CLICK)
