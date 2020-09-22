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

#' Main algorithm for attaining alpha, beta, C and D
#'
#' @param df training data consisting of userid, orderid and click. 
#' Ids should run continuously from 1 to end.
#' @param factors number of latent dimensions
#' @param lambda penalty term
#' @param iter maximum number of iterations
#' @param epsilon convergence limit
#' @param initType method of initialization
#' @param llh boolean for whether loglikelihood should be tracked
#' @param rmse boolean for whether RMSE should be tracked
#' @param df_test test set
#' @param a_in initial alpha (if warm start)
#' @param b_in initial beta (if warm start)
#' @param C_in initial C (if warm start)
#' @param D_in initial D (if warm start)
#'
#' @return returns parameters alpha, beta, C and D
#' 
parEst <- function(df, factors, lambda, iter, initType, llh, rmse, df_test=NULL, 
                   epsilon=NULL, a_in = NULL, b_in = NULL, C_in = NULL, D_in = NULL) {
  names(df)[1:3] <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  # Initialization (including centering)
  initPars <- initChoose(df, factors, lambda, initType, a_in, b_in, C_in, D_in)
  
  alpha <- initPars$alpha
  beta <- initPars$beta
  C <- initPars$C
  D <- initPars$D
  
  # Because Thijs' code uses matrix
  df <- as.matrix(df)
  
  nu <- max(df[,"USERID_ind"])
  ni <- max(df[,"OFFERID_ind"])
  
  #Retrieve indices for y=1 and y=0 from the input data
  y1 <- df[which(df[ ,"CLICK"] == 1), c("USERID_ind", "OFFERID_ind")]
  y0 <- df[which(df[ ,"CLICK"] == 0), c("USERID_ind", "OFFERID_ind")]
  
  df1 <- cbind(y1, "deriv" = NA)
  df0 <- cbind(y0, "deriv" = NA)
  
  gamma_y1 <- get_gamma0(y1[,1], y1[,2], alpha, beta, C, D)
  gamma_y0 <- get_gamma0(y0[,1], y0[,2], alpha, beta, C, D)
  
  # Initialize iteration number
  run <- 1
  
  if (!is.null(epsilon) || llh) {
    deviance_new <- sum(logllh1(gamma_y1)) + sum(logllh0(gamma_y0))
    objective_new <- deviance_new + 
      lambda/2 * norm(C, type="F")^2 + lambda/2 * norm(D, type="F")^2
  }
  
  if (llh) {
    # Keeping track of likelihoods
    deviance_all <- rep(NA, (iter+1))
    objective_all <- rep(NA, (iter+1))
    
    # Calculate log likelihood
    deviance_all[run] <- deviance_new
    objective_all[run] <- objective_new
  } else {
    deviance_all <- NA
    objective_all <- NA
  }
  
  if (rmse) {
    # Keeping track of rmse
    rmse_it <- rep(NA, (iter+1))
    pred <- getPredict(df_test, alpha, beta, C, D)
    predictions <- pred$prediction
    # Majority rule
    predictions[is.na(predictions)] <- 0
    actuals <- pred$CLICK
    
    rmse_it[run] <- sqrt(mean((predictions - actuals)^2))
  } else {
    rmse_it <- NA
  }
  
  # Keeping track of the number of factors
  factors_all <- rep(NA, (iter+1))
  factors_all[run] <- factors
  
  # Keeping track of parameters
  alpha_track <- alpha
  beta_track <- beta
  C_track <- C
  D_track <- D
  
  while (run <= iter) {
    # Keeping track of parameters
    alpha_track <- cbind(alpha_track, alpha)
    beta_track <- cbind(beta_track, beta)
    C_track <- cbind(C_track, C)
    D_track <- cbind(D_track, D)
    
    # tic(paste("Complete iteration", run, sep = " "))
    # Define low rank representation of gamma0
    low_rankC <- cbind(C, alpha, rep(1, nu))
    low_rankD <- cbind(D, rep(1,ni), beta)
    
    # Calculate gamma0
    # gamma0 <- low_rankC %*% t(low_rankD)
    
    # Calculate respective first derivatives for both y=1 and y=0
    df1[,"deriv"] <- -4 * derf1(gamma_y1)
    df0[,"deriv"] <- -4 * derf2(gamma_y0)
    
    # df1 <- cbind(y1, "deriv" = -4 * derf1(gamma0[y1]))
    # df0 <- cbind(y0, "deriv" = -4 * derf2(gamma0[y0]))
    
    # Combine the results in one matrix
    df01 <- rbind(df1, df0)
    
    # Create sparse matrix
    sparse <- sparseMatrix(i = df01[ ,"USERID_ind"], j = df01[ ,"OFFERID_ind"],
                           x = df01[ ,"deriv"], dims = c(nu, ni))
    
    # Calculating the H matrix for alpha update
    H_slr <- splr(sparse, low_rankC, low_rankD)
    
    # Updating alpha and beta
    newalpha <- rowMeans(H_slr)
    
    # Subtract the rowmean from H for the update for beta
    low_rankC <- cbind(C, (alpha - newalpha), rep(1, nu))
    H_slr_rowmean <- splr(sparse, low_rankC, low_rankD)
    newbeta <- colMeans(H_slr_rowmean)
    
    # Updating the C and D
    # Remove row and column mean from H
    low_rankD <- cbind(D, rep(1, ni), (beta - newbeta))
    H_slr_rowandcolmean <- splr(sparse, low_rankC, low_rankD)
    
    # Retrieve C and D from the svd.als function
    results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = lambda / 2)
    
    # Updates
    alpha <- newalpha
    beta <- newbeta
    # Using CD' = (UD^1/2)(VD^1/2)'
    
    # With one factor the ohter code gives an error due to d being scalar
    if (factors == 1) {
      C <- results$u %*% sqrt(results$d)
      D <- results$v %*% sqrt(results$d)
    }
    else {
      C <- results$u %*% diag(sqrt(results$d))
      D <- results$v %*% diag(sqrt(results$d))
    }
    
    run <- run + 1
    
    # Fooling around with parameters
    alpha <- runif(nu, min=-5, max=5)
    beta <- runif(ni, min=-5, max=5)
    C <- matrix(rnorm(nu * factors, 0, 5), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, 5), ni, factors)
    
    # Updating gamma
    gamma_y1 <- get_gamma0(y1[,1], y1[,2], alpha, beta, C, D)
    gamma_y0 <- get_gamma0(y0[,1], y0[,2], alpha, beta, C, D)
    
    if (!is.null(epsilon)) {
      deviance_old <- deviance_new
      objective_old <- objective_new
    }
    if (!is.null(epsilon) || llh) {
      deviance_new <- sum(logllh1(gamma_y1)) + sum(logllh0(gamma_y0))
      objective_new <- deviance_new +
        lambda / 2 * norm(C, type = "F") ^ 2 + lambda / 2 * norm(D, type = "F") ^ 2
    }
    
    if (llh){
      # Log Likelihood of current iteration
      deviance_all[run] <- deviance_new
      objective_all[run] <- objective_new
    }
    
    if (rmse){
      # RMSE of current iteration
      pred <- getPredict(df_test, alpha, beta, C, D)
      predictions <- pred$prediction
      predictions[is.na(predictions)] <- 0
      actuals <- pred$CLICK
      
      rmse_it[run] <- sqrt(mean((predictions - actuals)^2))
    }
    # toc()
    
    if (!is.null(epsilon)) {
      if (!is.infinite(objective_new) && !is.infinite(objective_old)) {
        print(paste("Iter", (run-1), "Change in deviance is", (deviance_new-deviance_old)/deviance_old, sep=" "))
        print((objective_new-objective_old)/objective_old)
        if (abs((objective_new-objective_old)/objective_old) < epsilon) break
      }
    }
    
    # Keeping track of the number of factors
    factors_all[run] <- sum(results$d > 0)
    
  }
  
  # Keeping track
  par_track <- list("alpha_track" = alpha_track, "beta_track" = beta_track, 
                    "C_track" = C_track, "D_track" = D_track)
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
                 "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all,
                 "par_track" = par_track)
  return(output)
}

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
iter <- 50
iter <- 100
initType <- 2
onlyVar <- TRUE
llh <- TRUE
rmse <- TRUE
epsilon <- 0.0000001
epsilon <- NULL

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
alpha_t <- runif(n, min=5, max=10)
# alpha_t <- alpha
beta <- runif(p, min=5, max=10)
#beta_t <- beta
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
df_test$CLICK <- probability_t[as.matrix(df_test[ , c("USERID", "OFFERID")])]
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
df_test <- df[as.logical(df$train_test), c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "ratioU", 
                                           "ratioO", "prediction")]
df_train <- df[!(as.logical(df$train_test)), c("USERID_ind_new", "OFFERID_ind_new", "CLICK", 
                                               "ratioU", "ratioO")]

# Getting predictions
set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)



### Overwrite simulation ----
# We can also simply overwrite the test clicks
set.seed(0)
df_test$CLICK <- sample(c(0,1), nrow(df_test), replace = TRUE)

# Getting predictions
set.seed(0)
output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)



