library(recosystem)
library(dplyr)
library(readr)
library(softImpute)
library(spam)
library(tidyverse)
library(tictoc)
library(bigmemory)
library(RcppArmadillo)

# Functions/tools ------------------------------------------------------------------------
f1 <- function(x){
  return(log(1+exp(-x)))
}

f2 <- function(x){
  return(x + log(1+exp(-x)))
}

f3 <- function(x) {
  return(0)
}

derf1 <- function(x){
  return(-1/(1+exp(x)))
}

derf2 <- function(x){
  return(1/(1+exp(-x)))
}

llh <- function(x){
  return(1 / (1 + exp(-x)))
}

logllh1 <- function(x){
  return(log(1+exp(-x)))
}

logllh0 <- function(x){
  return(x + log(1+exp(-x)))
}

trainTest <- function(df){
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  #1. Make the test train split (test is 1)
  df$train_test <- rbinom(n = nrow(df), size = 1, prob = 0.2)
  
  #2. Create new indices. Make sure test is at bottom
  df <- df[order(df$train_test, df$OFFERID_ind), ]
  df <- df %>% 
    mutate(OFFERID_indN = group_indices(., factor(OFFERID_ind, levels = unique(OFFERID_ind))))
  df <- df[order(df$train_test, df$USERID_ind), ]
  df <- df %>% 
    mutate(USERID_indN = group_indices(., factor(USERID_ind, levels = unique(USERID_ind))))
  
  #3. Split sets
  df_test <- df[as.logical(df$train_test), ]
  df_train <- df[!(as.logical(df$train_test)), ]
  
  #4. Return
  output <- list("df_train" = df_train, "df_test" = df_test)
  return(output)
}


initChoose <- function(df, factors, priorsdu, priorsdi, initType){
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  if (initType == 1){ 
    #Alpha and beta's initialized with a zero, C and D with normal priors
    df <- as.matrix(df)
    nu <- length(unique(df[ ,1]))
    ni <- length(unique(df[ ,2]))
    
    alpha <- rep(0, nu)
    beta <- rep(0, ni)
    C <- matrix(rnorm(nu * factors, 0, priorsdu), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, priorsdi), ni, factors)
  }
  else if (initType == 2) {
    nu <- max(df[ ,1])
    ni <- max(df[ ,2])
    avg <- mean(df$CLICK)
    gammaAvg <- -1 * log(1/avg - 1)
    
    gamma <- matrix(rnorm(nu * ni, 0, 1), nu, ni) + gammaAvg
    svdGamma <- svd(gamma, nu = 2, nv = 2)
    
    Ctemp <- svdGamma$u %*% diag(sqrt(svdGamma$d))
    Dtemp <- svdGamma$v %*% diag(sqrt(svdGamma$d))
    
    alpha <- rowMeans(Ctemp)
    beta  <- rowMenas(Dtemp)
    
    C <- Ctemp - matrix(rep(alpha, 2), nu, 2)
    D <- Dtemp - matrix(rep(beta, 2), ni, 2)
  }
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D)
  return(output)
}


#' Main algorithm for attaining alpha, beta, C and D
#'
#' @param df Dataframe consisting of userid, orderid and click. Id's should run continuously
#' from 1 to end.
#' @param factors "Width" of C and D
#' @param priorsdu Priors for variance for C?
#' @param priorsdi Priors for variance of D?
#' @param priorlambdau Prior for lambda for norm C
#' @param priorlambdai Prior for lambda for norm D
#' @param iter Iterlation limit
#'
#' @return returns parameters alpha, beta, C and D
parEst <- function(df, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, initType) {
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  tic("Initialization")
  # Initialization
  initPars <- initChoose(df, factors, priorsdu, priorsdi, initType)
  
  #Center required parameters for identification
  alpha <- initPars$alpha
  beta <- scale(initPars$beta, scale = FALSE)
  C <- scale(initPars$C, scale = FALSE)
  D <- scale(initPars$D, scale = FALSE)
  toc()
  
  # Because Thijs' code uses matrix
  df <- as.matrix(df)
  
  nu <- length(unique(df[ ,1]))
  ni <- length(unique(df[ ,2]))
  
  #Retrieve indices for y=1 and y=0 from the input data
  y1 <- df[which(df[ ,3] == 1), c("USERID_ind", "OFFERID_ind")]
  y0 <- df[which(df[ ,3] == 0), c("USERID_ind", "OFFERID_ind")]
  
  # Keeping track of likelihoods
  logllh <- rep(NA, (iter+1))
  
  run <- 0
  while (run <= iter) {
    tic(paste("Iteration", run, sep = " "))
    
    #Define low rank representation of gamma0
    low_rankC <- cbind(C, alpha, rep(1, nu))
    low_rankD <- cbind(D, rep(1,ni), beta)
    
    #Calculate gamma0
    gamma0 <- low_rankC %*% t(low_rankD)
    
    #Calculate respective first derivatives for both y=1 and y=0
    df1 <- cbind(y1, "deriv" = -4 * derf1(gamma0[y1]))
    df0 <- cbind(y0, "deriv" = -4 * derf2(gamma0[y0]))
    
    #Combine the results in one matrix
    df01 <- rbind(df1, df0)
    
    #Turn this matrix to sparse, notice that the dims had to be manually set (for missing items probably
    sparse <- sparseMatrix(i = (df01[ ,1]), j = (df01[ ,2]), x = df01[ ,3], dims = c(nu, ni))
    
    #Calculating the H matrix for alpha update
    H_slr <- splr(sparse, low_rankC, low_rankD)
    
    #Updating alpha and beta
    newalpha <- as.matrix((1/ni) * H_slr %*% rep(1, ni))
    
    #Subtract the rowmean from H for the update for beta
    low_rankC <- cbind(C, (alpha - rowMeans(H_slr)), rep(1, nu))
    H_slr_rowmean <- splr(sparse, low_rankC, low_rankD)
    newbeta <- as.matrix((1/nu) * t(t(rep(1, nu)) %*% H_slr_rowmean))
    
    #Updating the C and D
    #Remove row and column mean from H
    low_rankD <- cbind(D, rep(1, ni), (beta - colMeans(H_slr_rowmean)))
    H_slr_rowandcolmean <-splr(sparse, low_rankC, low_rankD)
    
    #Retrieve C and D from the svd.als function
    results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = priorlambdau / 2)
    
    # Updates
    alpha <- newalpha
    beta <- newbeta
    # Using CD' = (UD^1/2)(VD^1/2)'
    C <- results$u %*% diag(sqrt(results$d))
    D <- results$v %*% diag(sqrt(results$d))
    
    run <- run + 1
    
    tic("Calculating logllh")
    # Log Likelihood of PREVIOUS iteration
    logllh[run] <- sum(logllh1(gamma0[y1])) + sum(logllh0(gamma0[y0])) + 
      priorlambdau/2 * norm(C)^2 + priorlambdai/2 * norm(D)^2
    toc()
    
    
    toc()
  }
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "logllh" = logllh)
  return(output)
}






#' Get predictions for a test set
#'
#' @param df Test set consisting of USERID_ind and OFFERID_ind
#' @param alpha parameter estimate alpha
#' @param beta parameter estimate beta
#' @param C parameter estimate C
#' @param D parameter estimate D
#' @param uniqueU basically a vector used as a dictionary
#' @param uniqueI basically a vector used as a dictionary
#'
#' @return dataframe including predictions, NA for unknown user/item
getPredict <- function(df, alpha, beta, C, D){
  tic("Prediction")
  
  # By using the size of C and D, we can infer which obs are missing in training
  maxU <- nrow(C)
  maxI <- nrow(D)
  
  # Use low rank (because it seemed fast) and calculate gamma
  low_rankC <- cbind(C, alpha, rep(1, nrow(C)))
  low_rankD <- cbind(D, rep(1, nrow(D)), beta)
  gamma <- low_rankC %*% t(low_rankD)
  
  # Marking offer/items that are non existent
  df$nonMiss <- ((df[ ,1]  <= maxU) & (df[ ,2] <= maxI))
  
  # Predicting for the non missing ones (NA for non missings)
  df$prediction <- NA
  # Get the non missing indices
  nonMiss <- as.matrix(df[df$nonMiss, c(1, 2)])
  df$prediction[df$nonMiss] <- llh(gamma[nonMiss])
  
  # Predicting for the missing ones
  #df$prediction[!df$nonMiss] <- NA
  
  toc()
  return(df)
}

fullAlg <- function(df, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, initType){
  tic("Total time")
  # Train test split
  tic("1. Train test split")
  df_train <- trainTest(df)$df_train[ ,c("USERID_indN", "OFFERID_indN", "CLICK")]
  df_test <- trainTest(df)$df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK")]
  toc()
  
  # Estimating parameters
  tic("2. Estimating parameters")
  pars <- parEst(df_train, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, initType)
  toc()
  
  # Getting predictions
  tic("3. Getting predictions")
  results <- getPredict(df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK")], 
                        pars$alpha, pars$beta, pars$C, pars$D)
  toc()
  
  # RMSE
  results$prediction[is.na(results$prediction)] <- 0
  RMSE <- sqrt(mean((results$prediction - results$CLICK)^2))
  
  # Output
  output <- list("parameters" = pars, "prediction" = results, "RMSE" = RMSE)
  toc()
  return(output)
}





# Preparing data -------------------------------------------------------------------------
# This is how you should import the data.
# The sequence here is important. We want to have a continuous sequence, starting at 1
# for the indices for user and order in our training set.

#1. Import train and test set
df_train <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/Observations_Report.csv",
                       ";", escape_double = FALSE, trim_ws = TRUE)
df_test <- read_delim("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/Observations_Game.csv",
                      ";", escape_double = FALSE, trim_ws = TRUE)

#2. Create indices 
df_test$CLICK <- NA
df <- rbind(df_train, df_test)

# Order on click first such that NA are at bottom (no missings indices in training data)
df <- df[order(df$CLICK, df$OFFERID), ]
df <- df %>% 
  mutate(OFFERID_ind = group_indices(., factor(OFFERID, levels = unique(OFFERID))))
df <- df[order(df$CLICK, df$USERID), ]
df <- df %>% 
  mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))

#3. Split sets and save test
df_test <- df[is.na(df$CLICK), c("USERID_ind", "OFFERID_ind")]
df_train <- df[!(is.na(df$CLICK)), ]
saveRDS(df_test, "/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_test")

#4. Remove duplicates by keeping latest observation
df_train <- df_train[order(df_train$MAILID, decreasing=TRUE), ]
df_train <- df_train[!duplicated(df_train[c("USERID", "OFFERID")]), c("USERID_ind", "OFFERID_ind", "CLICK")]

#5.Save train
saveRDS(df_train, "/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")



# Train/test predictions -----------------------------------------------------------------
# Use "Preparing data" first to get the df_train object
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")

# Setting parameters
factors <- 2
priorsdu <- 2.5
priorsdi <- 2.5
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi
iter <- 0
initType <- 1

output <- fullAlg(df, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, initType)

# Visualization
hist(output$prediction$prediction)
plot(output$parameters$logllh)

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

max(df_train$USERID_indN)
length(unique(df_train$USERID_indN))

max(df_train$OFFERID_indN)
length(unique(df_train$OFFERID_indN))

# General parameter estimation algorithm testing
factors <- 2
priorsdu <- 2.5
priorsdi <- 2.5
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi
iter <- 0
initType <- 1

pars <- parEst(parEst(df_train, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, initType))
war