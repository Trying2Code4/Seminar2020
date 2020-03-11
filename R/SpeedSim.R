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

makeData <- function(nu, ni, sparsity, f=3, alpha = NULL, beta=NULL, C=NULL, D=NULL){
  # If not predetermined udnerlying model is given
  if (is.null(alpha)){
    alpha <- runif(nu, min=-5, max=0)
  }
  if (is.null(beta)){
    beta <- runif(ni, min=-5, max=0)
  }
  if (is.null(C)){
    C <- matrix(rnorm(nu * f, 0, 2), nu, f)
  }
  if (is.null(D)){
    D <- matrix(rnorm(ni * f, 0, 2), ni, f)
  }

  
  gamma <- alpha + beta + C %*% t(D) + matrix(rnorm(nu*ni, 0, 1), nu, ni)
  probability <- exp(gamma) / (1 + exp(gamma))
  
  # Create a train subset with a certain sparsity level
  sparsity <- sparsity
  
  # Retrieve random elements from the matrix
  USERID <- rep(c(1:nu), each = ni)
  OFFERID <- rep(c(1:ni), times = nu)
  df <- data.frame("USERID" = USERID, "OFFERID" = OFFERID)
  df <- df[sample(1:length(USERID), size = sparsity * nu * ni), ]
  
  # And the corresponding clicks
  df$CLICK <- probability[as.matrix(df[ , c("USERID", "OFFERID")])]
  df$CLICK <- as.numeric(df$CLICK > 0.5)
  
  return(df)
}

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
  
  runtime <- rep(NA, iter)
  
  while (run <= iter) {
    time <- system.time({
    
    
    tic(paste("Complete iteration", run, sep = " "))
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
    results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = lambda * 4)
    
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
    toc()
    
    if (!is.null(epsilon)) {
      if (!is.infinite(objective_new) && !is.infinite(objective_old)) {
        print(paste("Iter", (run-1), "Change in deviance is", (deviance_new-deviance_old)/deviance_old, sep=" "))
        print((objective_new-objective_old)/objective_old)
        if (abs((objective_new-objective_old)/objective_old) < epsilon) break
      }
    }
    
    # Keeping track of the number of factors
    factors_all[run] <- sum(results$d > 0)
    
    })
    runtime[run-1] <- time
    
  }
  
  # Keeping track
  par_track <- list("alpha_track" = alpha_track, "beta_track" = beta_track, 
                    "C_track" = C_track, "D_track" = D_track)
  
  meanTime <- mean(runtime)
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
                 "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all,
                 "par_track" = par_track, "meanTime" = meanTime)
  return(output)
}

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
parEstSlow <- function(df, factors, lambda, iter, initType, llh, rmse, df_test=NULL, 
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
    
    
    
    tic(paste("Complete iteration", run, sep = " "))
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
    
    # +++++++++++++ THIS CHAHNGED NOW ++++++++++++++++
    H_slr <- sparse + low_rankC%*%t(low_rankD)
    
    # Updating alpha and beta
    newalpha <- rowMeans(H_slr)
    
    # Subtract the rowmean from H for the update for beta
    H_slr_rowmean <- t(scale(t(H_slr), scale = FALSE))
    newbeta <- colMeans(H_slr_rowmean)
    
    # Updating the C and D
    # Remove row and column mean from H
    H_slr_rowandcolmean <- scale(H_slr_rowmean, scale = FALSE)
    
    # Retrieve C and D from the svd.als function
    results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = lambda * 4)
    
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
    toc()
    
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

<<<<<<< HEAD
testdata <- makeData(100,100,0.5, 5)
parEst(testdata, 3, 10, 100, 2, FALSE, FALSE)
parEstSlow(testdata, 3, 10, 100, 2, FALSE, FALSE)
testdata <- df_train[df_train$USERID_ind < 1000,]

=======
tm1 <- system.time(
  {
    sample(1:100000000, size=100000)
  })

tm1[1] - 0.1
>>>>>>> c6cfc71248e3bc37a8a02c9135ee922a6ddbf84c

  