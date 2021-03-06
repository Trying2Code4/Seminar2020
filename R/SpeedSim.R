library(dplyr)
library(readr)
library(softImpute)
library(tidyverse)
library(tictoc)
library(RcppArmadillo)
library(Rcpp)
library(ggplot2)
library(openxlsx)
library(reshape2)
require(MASS) # to access Animals data sets
require(scales)
library(gridExtra)


# sourceCpp("/Users/colinhuliselan/Documents/Master/Seminar/Seminar2020_V2/R/gammaui.cpp")
#sourceCpp("~/Dropbox/Uni/Master_Econometrie/Blok_3/Seminar2020/R/gammaui.cpp")
sourceCpp("gammaui.cpp")
source("MajorizationFunctions.R")

norm_vec <- function(x) {sqrt(sum(x^2))}

makeData <- function(nu, ni, sparsity, f=2, alpha = NULL, beta=NULL, C=NULL, D=NULL,
                     model=T){
  
  # Create a train subset with a certain sparsity level
  # Make indice matrices
  # First we take something like a "diagonal"
  USERID <- rep(c(1:nu), times= ni)
  OFFERID <- rep(c(1:ni), times = nu)
  df <- data.frame("USERID" = USERID, "OFFERID" = OFFERID)
  df_diag <- df[1:max(nu,ni), ]
  
  # Sample from the remainder
  USERID <- rep(c(1:nu), each = ni)
  OFFERID <- rep(c(1:ni), times = nu)
  df <- data.frame("USERID" = USERID, "OFFERID" = OFFERID)
  df <- anti_join(df, df_diag)
  
  df_nodiag <- df[sample(1:nrow(df), size = (sparsity * nu * ni - nrow(df_diag))), ]
  
  # Combine diag and remainder
  df <- rbind(df_diag, df_nodiag)
  
  if (model){
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
    
    low_rankC <- cbind(C, alpha, rep(1, nu))
    low_rankD <- cbind(D, rep(1,ni), beta)
    
    gamma <- low_rankC %*% t(low_rankD) + matrix(rnorm(nu*ni, 0, 1), nu, ni)
    probability <- exp(gamma) / (1 + exp(gamma))
    
    # And the corresponding clicks
    df$CLICK <- probability[as.matrix(df[ , c("USERID", "OFFERID")])]
    df$CLICK <- as.numeric(df$CLICK > 0.5)
  } else{
    df$CLICK <- rbinom(n = nrow(df), size = 1, prob = 0.05)
  }
  
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
                   epsilon=NULL, a_in = NULL, b_in = NULL, C_in = NULL, D_in = NULL,
                   gradient = FALSE, ...) {
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
  
  if ((!is.null(epsilon) & !gradient) || llh) {
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
  
  if (gradient) {
    gradient_all <- matrix(nrow = 4, ncol = iter)
  } else {
    gradient_all <- NA
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
    results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = lambda * 4, ...)
    
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
    
    if (gradient) {
      normParam <- normGrad(df, lambda, alpha, beta, C, D)
      gradient_all[, run-1] <- normParam
    }
    
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
    runtime[run-1] <- time[3]
    
  }
  
  # Keeping track
  par_track <- list("alpha_track" = alpha_track, "beta_track" = beta_track, 
                    "C_track" = C_track, "D_track" = D_track)
  
  meanTime <- mean(runtime[-1])
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
                 "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all,
                 "par_track" = par_track, "meanTime" = meanTime, "gradient" = gradient_all)
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
  
  runtime <- rep(NA, iter)
  
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
    
    # +++++++++++++ THIS CHAHNGED NOW ++++++++++++++++
    # Calculating the H matrix for alpha update
    H_slr <- sparse + low_rankC%*%t(low_rankD)
    
    # Updating alpha and beta
    newalpha <- rowMeans(H_slr)
    
    # Subtract the rowmean from H for the update for beta
    # H_slr_rowmean <- t(scale(t(H_slr), scale = FALSE))
    H_slr_rowmean <- H_slr - newalpha
    newbeta <- colMeans(H_slr_rowmean)
    
    # Updating the C and D
    # Remove row and column mean from H
    # H_slr_rowandcolmean <- scale(H_slr_rowmean, scale = FALSE)
    H_slr_rowandcolmean <- t(t(H_slr_rowmean) - newbeta)
    H_slr_rowandcolmean <- as(H_slr_rowandcolmean, "sparseMatrix")
    # a <- matrix(0, nrow=nu, ncol=1)
    # b <- matrix(0, nrow=ni, ncol=1)
    # H_slr_rowandcolmean <- splr(H_slr_rowandcolmean)
    
    # Retrieve C and D from the svd.als function
    results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = lambda * 4, 
                       final.svd = FALSE)
    
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
    runtime[run-1] <- time[3]
    
  }
  
  # Keeping track
  par_track <- list("alpha_track" = alpha_track, "beta_track" = beta_track, 
                    "C_track" = C_track, "D_track" = D_track)
  
  meanTime <- mean(runtime[-1])
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
                 "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all,
                 "par_track" = par_track, "meanTime" = meanTime)
  return(output)
}

normGrad <- function(df, lambda, alpha, beta, C, D, prop = 0.1) {
  # Take sample of the data
  # df_small <- data.frame(df[sample(nrow(df), floor(prop*nrow(df)), replace = FALSE), c("USERID_ind", "OFFERID_ind", "CLICK")])
  df <- as.data.frame(df)
  colnames(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  # Take random sample of users
  df_small <- data.frame(df[df$USERID_ind %in% sample(max(df$USERID_ind), floor(prop*max(df$USERID_ind)), replace = FALSE), c("USERID_ind", "OFFERID_ind", "CLICK")])
  
  gamma <- get_gamma0(df_small$USERID_ind, df_small$OFFERID_ind, alpha, beta, C, D)
  df_small$difference <- df_small$CLICK - mu(gamma)
  
  # Get the gradients of alpha
  deriv_alpha <- df_small %>% group_by(USERID_ind) %>% summarise(deriv_alpha = sum(difference)) %>% dplyr::select(deriv_alpha) %>% ungroup()
  
  # Get gradient of C (not vectorized, but that doesn't matter since we take the norm anyway)
  mult_C <- tcrossprod(df_small$difference, rep(1, ncol(D))) * D[df_small$OFFERID_ind,]
  mult_C <- data.frame(cbind("USERID_ind" = df_small$USERID_ind, mult_C))
  deriv_C <- mult_C %>% group_by(USERID_ind) %>% summarise_all(sum) %>% ungroup()
  deriv_C <- as.matrix(deriv_C[,-1] - lambda*C[deriv_C$USERID_ind, ])
  
  # Take random sample of offers
  df_small2 <- data.frame(df[df$OFFERID_ind %in% sample(max(df$OFFERID_ind), floor(prop*max(df$OFFERID_ind)), replace = FALSE), c("USERID_ind", "OFFERID_ind", "CLICK")])
  
  gamma2 <- get_gamma0(df_small2$USERID_ind, df_small2$OFFERID_ind, alpha, beta, C, D)
  df_small2$difference <- df_small2$CLICK - mu(gamma2)
  
  # Get the gradients of beta
  deriv_beta <- df_small2 %>% group_by(OFFERID_ind) %>% summarise(deriv_beta = sum(difference)) %>% dplyr::select(deriv_beta) %>% ungroup()
  
  # Get gradient of D (not vectorized, but that doesn't matter since we take the norm anyway)
  mult_D <- tcrossprod(df_small2$difference, rep(1, ncol(C))) * C[df_small2$USERID_ind,]
  mult_D <- data.frame(cbind("OFFERID_ind" = df_small2$OFFERID_ind, mult_D))
  deriv_D <- mult_D %>% group_by(OFFERID_ind) %>% summarise_all(sum) %>% ungroup()
  deriv_D <- as.matrix(deriv_D[,-1] - lambda*D[deriv_D$OFFERID_ind, ])
  
  return(c("norm_alpha" = norm_vec(deriv_alpha), "norm_beta" = norm_vec(deriv_beta), "norm_C" = norm(deriv_C, type="F"), "norm_D" = norm(deriv_D, type="F")))
}

normFullGrad <- function(df, lambda, alpha, beta, C, D) {
  df_small <- data.frame(df[, c("USERID_ind", "OFFERID_ind", "CLICK")])
  
  gamma <- get_gamma0(df_small$USERID_ind, df_small$OFFERID_ind, alpha, beta, C, D)
  df_small$difference <- df_small$CLICK - mu(gamma)
  
  # Get the gradients of alpha and beta
  deriv_alpha <- df_small %>% group_by(USERID_ind) %>% summarise(deriv_alpha = sum(difference)) %>% dplyr::select(deriv_alpha) %>% ungroup()
  deriv_beta <- df_small %>% group_by(OFFERID_ind) %>% summarise(deriv_beta = sum(difference)) %>% dplyr::select(deriv_beta) %>% ungroup()
  
  # Get gradient of C (not vectorized, but that doesn't matter since we take the norm anyway)
  mult_C <- tcrossprod(df_small$difference, rep(1, ncol(D))) * D[df_small$OFFERID_ind,]
  mult_C <- data.frame(cbind("USERID_ind" = df_small$USERID_ind, mult_C))
  deriv_C <- mult_C %>% group_by(USERID_ind) %>% summarise_all(sum) %>% ungroup()
  deriv_C <- as.matrix(deriv_C[,-1] - lambda*C[deriv_C$USERID_ind, ])
  
  # Get gradient of D (not vectorized, but that doesn't matter since we take the norm anyway)
  mult_D <- tcrossprod(df_small$difference, rep(1, ncol(C))) * C[df_small$USERID_ind,]
  mult_D <- data.frame(cbind("OFFERID_ind" = df_small$OFFERID_ind, mult_D))
  deriv_D <- mult_D %>% group_by(OFFERID_ind) %>% summarise_all(sum) %>% ungroup()
  deriv_D <- as.matrix(deriv_D[,-1] - lambda*D[deriv_D$OFFERID_ind, ])
  
  return(c("norm_alpha" = norm_vec(deriv_alpha), "norm_beta" = norm_vec(deriv_beta), "norm_C" = norm(deriv_C, type="F"), "norm_D" = norm(deriv_D, type="F")))
}


speedSim <- function(NU, NI, SPARSITY, FACTORS, file="speedSim.xlsx"){
  # Initialize a multidimensional output array
  # Rows are all the possible permutations of the huperparameters * folds
  rows <- (length(NU) * length(NI) * length(SPARSITY) * length(FACTORS))
  
  # Columns for the hyperparameters, plus a name variable, and then all the results you want
  # these are: mean time for both methods
  columns <- 4 + 1 + 2
  
  # Initialize the df (depth is the number of folds)
  output <- data.frame(matrix(NA, nrow = rows, ncol = columns))
  names(output) <- c("nu", "ni", "sparsity", "factors", "Specification",
                     "meanTimeFast", "meanTimeSlow")
  
  row <- 1
  
  for (a in 1:length(NU)){
    for (b in 1:length(NI)){
      for (c in 1:length(SPARSITY)){
        for (d in 1:length(FACTORS)){
          tic(paste("Run", row, "out of", rows))
          
          nu <- NU[a]
          ni <- NI[b]
          sparsity <- SPARSITY[c]
          factors <- FACTORS[d]
          
          # Create the data
          df <- makeData(nu, ni, sparsity)
          
          # Run the algorithms
          lambda <- 1
          iter <- 6
          initType <- 2
          onlyVar <- T
          llh <- FALSE
          rmse <- FALSE
          epsilon <- 0.001
          
          fast <- parEst(df, factors, lambda, iter, initType, llh, rmse, epsilon=epsilon)
          
          slow <- parEstSlow(df, factors, lambda, iter, initType, llh, rmse, epsilon=epsilon)
         
          # Fill the output
          output$nu[row] <- nu
          output$ni[row] <- ni
          output$sparsity[row] <- sparsity
          output$factors[row] <- factors
          output$Specification[row] <- paste("nu = ", nu, ", ni = ", ni, 
                                        ", sparsity = ", sparsity, ", factors = ", factors,
                                        sep = "")
          output$meanTimeFast[row] <- fast$meanTime
          output$meanTimeSlow[row] <- slow$meanTime
          
          write.xlsx(output, file = file)
          
          row <- row+1
          
          toc()
          
        }
      }
    }
  }
  return(output)
}


#' Function for comparing speeds of estimating parameters on different subset of data
#'
#' @param df entire dataframe
#' @param subsets list of subset sizes used
#' @param ... parameters for parEst
#'
#' @return
compareSpeed <- function(df, subsets, FACTORS, LAMBDA, ...) {
  total <- data.frame(matrix(nrow = length(subsets)*length(FACTORS)*length(LAMBDA), ncol = 4))
  colnames(total) <- c("subset", "factor", "lambda", "meanTime")
  
  row <- 1
  for (i in subsets) { 
    df_sub <- df[c(1:i), c("USERID_ind", "MailOffer", "CLICK")]
    df_sub <- df_sub %>% 
      mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))
    
    for (f in FACTORS) {
      for (l in LAMBDA) {
        total$subset[row] <- i
        total$factor[row] <- f
        total$lambda[row] <- l
        total$meanTime[row] = parEst(df_sub[, c("USERID_ind", "OFFERID_ind", "CLICK")], factors = f, lambda = l, ...)$meanTime
        row = row + 1
      }
    }
  }
  total <- total[order(total$subset, total$factor, total$lambda)]
  return(total)
}

#### Run it ------------------------------------------------------------------------------

debug(makeData)
# Create data
tic("hard model")
df <- makeData(nu=10^6, ni=100, sparsity=0.05, model=T)
toc()

tic("simple model")
df <- makeData(nu=10^6, ni=100, sparsity=0.05, model=F)
toc()

# Or load data
df <- readRDS("df_train.RDS")
df <- df[, c("USERID", "MailOffer", "CLICK")]
split <- trainTest(df, onlyVar)
df <- split$df_train[ ,c("USERID_ind", "OFFERID_ind", "CLICK")]

length(unique(df$USERID))
length(unique(df$OFFERID))
nrow(unique(df))


# Run the algorithms
factors <- 5
lambda <- 1
iter <- 5
initType <- 2
onlyVar <- T
llh <- FALSE
rmse <- FALSE
epsilon <- 0.00001


set.seed(123)
fast <- parEst(df, factors, lambda, iter, initType, llh, rmse, epsilon=epsilon)

set.seed(123)
slow <- parEstSlow(df, factors, lambda, iter, initType, llh, rmse, epsilon=epsilon)

fast$alpha - slow$alpha

debug(parEstSlow)



#### Run the simulation ------------------------------------------------------------------
NU <- round(c(10^2, 10^2.5, 10^3, 10^3.5, 10^4, 10^4.5, 10^5, 10^5.5, 10^6))
NI <- c(100)
SPARSITY <- c(0.05, 0.25)
FACTORS <- c(5, 20)

speedsim1 <- speedSim(NU, NI, SPARSITY, FACTORS, file="speedsim1.xlsx")
speedsim1 <- read.xlsx("speedsim1.xlsx")


#### Create output -----------------------------------------------------------------------
# Create a percentage difference
speedsim1$diff <- speedsim1$meanTimeSlow/speedsim1$meanTimeFast

# Making a figures for runs with 5 factors
dftemp_5 <- speedsim1[speedsim1$factors == 5, ]
dftemp_5_abs <- melt(dftemp_5, id = c("nu", "sparsity") , measure = c("meanTimeFast", "meanTimeSlow"))

# The plots for the actual comp times
cols <- c("meanTimeFast" = "black", "meanTimeSlow" = "#bfbdbd")
f5_abs1 <- ggplot(dftemp_5_abs, aes(x=nu, y=value, group=interaction(variable, sparsity),
                                colour = variable, linetype = factor(sparsity)))+
  geom_line()+
  geom_point()+
  scale_color_manual(name="Matrix type", labels=c("Sparse + LR", "Full matrix"),
                     values=cols)+
  scale_linetype_discrete(name="Sparsity level", labels=c("95%", "75%"))+
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(x ="Number of users", y = "Mean iteration time in sec.")+
  theme_bw()+
  theme(legend.position = c(0.3, 0.6))

f5_abs2 <- f5_abs1 +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))+
  labs(x ="Number of users", y = "Mean iteration time in sec. (log scale)")+
  theme(legend.position = "none")

# Plot for the relative computation time
dftemp_5_rel <- speedsim1[speedsim1$factors == 5, ]
f5_rel <- ggplot(dftemp_5_rel, aes(x=nu, y=diff, group=sparsity, linetype = factor(sparsity)))+
  geom_line()+
  geom_point()+
  ylim(0.5, 5)+
  scale_linetype_discrete(name="Sparsity level", labels=c("95%", "75%"))+
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(x ="Number of users", y = "Ratio of mean iteration times")+
  theme_bw()+
  theme(legend.position = "none")


# Making a figures for runs with 20 factors
dftemp_20 <- speedsim1[speedsim1$factors == 20, ]
dftemp_20_abs <- melt(dftemp_20, id = c("nu", "sparsity") , measure = c("meanTimeFast", "meanTimeSlow"))

# The plots for the actual comp times
cols <- c("meanTimeFast" = "black", "meanTimeSlow" = "#bfbdbd")
f20_abs1 <- ggplot(dftemp_20_abs, aes(x=nu, y=value, group=interaction(variable, sparsity),
                                    colour = variable, linetype = factor(sparsity)))+
  geom_line()+
  geom_point()+
  scale_color_manual(name="Matrix type", labels=c("Sparse + LR", "Full matrix"),
                     values=cols)+
  scale_linetype_discrete(name="Sparsity level", labels=c("95%", "75%"))+
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(x ="Number of users", y = "Mean iteration time in sec.")+
  theme_bw()+
  theme(legend.position = c(0.3, 0.6))

f20_abs2 <- f20_abs1 +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(x ="Number of users", y = "Mean iteration time in sec. (log scale)")+
  theme(legend.position = "none")

# Plot for the relative computation time
dftemp_20_rel <- speedsim1[speedsim1$factors == 20, ]
f20_rel <- ggplot(dftemp_20_rel, aes(x=nu, y=diff, group=sparsity, linetype = factor(sparsity)))+
  geom_line()+
  geom_point()+
  ylim(0.5, 5)+
  scale_linetype_discrete(name="Sparsity level", labels=c("95%", "75%"))+
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  labs(x ="Number of users", y = "Ratio of mean iteration times")+
  theme_bw()+
  theme(legend.position = "none")


# Arranging in a grid
grid.arrange(f5_abs1, f5_abs2, f5_rel, ncol = 3, nrow = 1)
grid.arrange(f20_abs1, f20_abs2, f20_rel, ncol = 3, nrow = 1)


#### Compare speeds ----------------------------------------------------------------------

# Disable scientific notation
options(scipen=999)

df <- readRDS("df_obs.RDS")
df <- df[order(df$USERID, df$MailOffer), c("USERID", "MailOffer", "CLICK")]
df <- df %>% mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))

proportions <- c(0.2, 0.4, 0.6, 0.8, 1)
subsets <- floor(proportions*nrow(df))

subset <- subsets[1]
FACTORS <- c(5, 10)
LAMBDA <- c(5, 10)
iter <- 6 # So we still have the average over 5 iterations after dropping the first one
initType <- 2
llh <- FALSE
rmse <- FALSE

# compareSpeed uses the parEst in this file!
speedTest <- compareSpeed(df, subset, FACTORS, LAMBDA, iter, initType, llh, rmse)

# Algorithm using gradients --------------------------------------------------------------

# Full grad vs MM ------------------------------------------------------------------------
# Import full observation set

df_obs <- readRDS("df_obs.RDS")
df_obs <- df_obs[ , c("USERID_ind", "OFFERID_ind", )]

factors <- 10
lambda <- 10
iter <- 5
initType <- 2
onlyVar <- T
llh <- FALSE
rmse <- FALSE
epsilon <- 0.00001


set.seed(123)
fast <- parEst(df_obs, factors, lambda, iter, initType, llh, rmse, epsilon=epsilon)

# Checking gradients for previous output ---------------------------------------------------

# 1.
df_train <- readRDS("Data/df_train")
df_train <- df_train[, c("USERID", "MailOffer", "CLICK")]

df_val <- readRDS("Data/df_val")
df_val <- df_val[, c("USERID", "MailOffer", "CLICK")]

prep <- prepData(df_train, df_val, onlyVar = TRUE)
train <- prep$df_train

parEst_f5_l5 <- readRDS("parEst/parEst_f5_l5.RDS")

grad_f5_l5 <- matrix(nrow = 5, ncol = 4)
for (i in 1:nrow(grad_f10_l10)) {
  grad_f5_l5[i,] <- normGrad(train, parEst_f5_l5$lambda, parEst_f5_l5$parEst$alpha, parEst_f5_l5$parEst$beta, parEst_f5_l5$parEst$C, parEst_f5_l5$parEst$D)
}

# 2.
df_obs <- readRDS("Data/df_obs.RDS")
df_train <- df_obs[df_obs$res == 0, c("USERID", "MailOffer", "CLICK")]
df_res <- df_obs[df_obs$res == 1, c("USERID", "MailOffer", "CLICK")]

prep <- prepData(df_train, df_res, onlyVar = FALSE)
train <- prep$df_train

outputRes <- readRDS("outputRes.RDS")

grad_f10_l10 <- matrix(nrow = 5, ncol = 4)
for (i in 1:nrow(grad_f10_l10)) {
  grad_f10_l10[i,] <- normGrad(train, lambda = 10, outputRes$parameters$alpha, outputRes$parameters$beta, outputRes$parameters$C, outputRes$parameters$D)
}
  

# Algorithm using gradients ----------------------------------------------------------------

factors <- 5
lambda <- 5
iter <- 250
initType <- 2
epsilon <- 1e-06

df_train <- readRDS("Data/df_train")
df_train <- df_train[, c("USERID", "MailOffer", "CLICK")]

df_val <- readRDS("Data/df_val")
df_val <- df_val[, c("USERID", "MailOffer", "CLICK")]

prep <- prepData(df_train, df_val, onlyVar = TRUE)
train <- prep$df_train[, c("USERID_ind", "OFFERID_ind", "CLICK")]

# Estimate parameters
set.seed(0)
MM_grad <- parEst(train, factors, lambda, iter, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon, gradient=TRUE, final.svd = TRUE)

set.seed(0)
MM_grad_highlambda <- parEst(train, factors, lambda = 50, iter = 100, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon, gradient=TRUE, final.svd = TRUE)

set.seed(0)
MM_grad_nofinal <- parEst(train, factors, lambda, iter, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon, gradient=TRUE, final.svd = FALSE)

par(mfrow=c(2,2))
plot(MM_grad$gradient[1,],
     col="blue", type = "l", lwd=2, ylab="Gradient of alpha", xlab="Iteration")
plot(MM_grad$gradient[2,],
     col="green", type = "l", lwd=2, ylab="Gradient of beta", xlab="Iteration")
plot(MM_grad$gradient[3,],
     col="red", type = "l", lwd=2, ylab="Gradient of C", xlab="Iteration")
plot(MM_grad$gradient[4,],
     col="orange", type = "l", lwd=2, ylab="Gradient of D", xlab="Iteration")
par(mfrow=c(1,1))

# Finding norm for a small dataset -----------------------------------------------------

df_train <- readRDS("Data/df_train")
df_train <- df_train[order(df_train$USERID, df_train$MailOffer), c("USERID", "MailOffer", "CLICK")]
df <- df_train[c(1:1000000),]

df <- df %>% 
  mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))
df <- df %>% 
  mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))

factors <- 5
lambda <- 5
iter <- 1000
initType <- 2
epsilon <- 1e-06

parEst_grad <- parEst(df[, c("USERID_ind", "OFFERID_ind", "CLICK")], factors, lambda, iter, initType, llh=TRUE, rmse=FALSE, epsilon=epsilon, gradient=TRUE)

par(mfrow=c(2,2))
plot(parEst_grad$gradient[1,c(1:parEst_grad$run-1)],
     col="blue", type = "l", lwd=2, ylab="Gradient of alpha", xlab="Iteration")
plot(parEst_grad$gradient[2,c(1:parEst_grad$run-1)],
     col="green", type = "l", lwd=2, ylab="Gradient of beta", xlab="Iteration")
plot(parEst_grad$gradient[3,c(1:parEst_grad$run-1)],
     col="red", type = "l", lwd=2, ylab="Gradient of C", xlab="Iteration")
plot(parEst_grad$gradient[4,c(1:parEst_grad$run-1)],
     col="orange", type = "l", lwd=2, ylab="Gradient of D", xlab="Iteration")
par(mfrow=c(1,1))

alpha <- parEst_grad$alpha
beta <- parEst_grad$beta
C <- parEst_grad$C
D <- parEst_grad$D

gradients <- parEst_grad$gradient[,c(1:parEst_grad$run - 1)]

write.csv(alpha, file = "alpha_small.csv")
write.csv(beta, file = "beta_small.csv")
write.csv(C, file = "C_small.csv")
write.csv(D, file = "D_small.csv")
write.csv(gradients, file = "gradient_small.csv")

# calculate full gradient
partialGrad <- normGrad(df, 5, alpha, beta, C, D)
fullGrad <- normFullGrad(df, 5, alpha, beta, C, D)

# Comparing gradients (10^4) -------------------------------------------------------------------
df_obs10k <- read_delim("Data/DataGradient/df_obs10k.csv", ",", escape_double = FALSE, trim_ws = TRUE)
df_obs10k <- df_obs10k[,c("USERID_ind", "OFFERID_ind", "CLICK")]
pars10k <- readRDS("ParEst/pars10k.RDS")

set.seed(0)
normGrad10k <- normGrad(df_obs10k, lambda = 5, pars10k$alpha, pars10k$beta, pars10k$C, pars10k$D)
normFullGrad10k <- normFullGrad(df_obs10k, lambda = 5, pars10k$alpha, pars10k$beta, pars10k$C, pars10k$D)

# Comparing gradients (10^(4.5)) -------------------------------------------------------------------
df_obs50k <- read_delim("Data/DataGradient/df_obs50k.csv", ",", escape_double = FALSE, trim_ws = TRUE)
df_obs50k <- df_obs50k[,c("USERID_ind", "OFFERID_ind", "CLICK")]
pars50k <- readRDS("ParEst/pars50k.RDS")

set.seed(0)
normGrad50k <- normGrad(df_obs50k, lambda = 5, pars50k$alpha, pars50k$beta, pars50k$C, pars50k$D)
normFullGrad50k <- normFullGrad(df_obs50k, lambda = 5, pars50k$alpha, pars50k$beta, pars50k$C, pars50k$D)

# Comparing gradients (10^5) -------------------------------------------------------------------
df_obs100k <- read_delim("Data/DataGradient/df_obs100k.csv", ",", escape_double = FALSE, trim_ws = TRUE)
df_obs100k <- df_obs100k[,c("USERID_ind", "OFFERID_ind", "CLICK")]
pars100k <- readRDS("ParEst/pars100k.RDS")

set.seed(0)
normGrad100k <- normGrad(df_obs100k, lambda = 5, pars100k$alpha, pars100k$beta, pars100k$C, pars100k$D)
normFullGrad100k <- normFullGrad(df_obs100k, lambda = 5, pars100k$alpha, pars100k$beta, pars100k$C, pars100k$D)

# Output images gradients and llh -------------------------------------------------------------------

# CAUTION: uses a different df_obs10k dataset than above
set.seed(0)
pars10k_grad <- parEst(df_obs10k, factors = 10, lambda = 5, iter = 1000, initType = 2, llh = TRUE, 
                       rmse = FALSE, df_test=NULL, epsilon = 1e-06, gradient = TRUE)

gradients <- data.frame(t(pars10k_grad$gradient[, c(1:pars10k_grad$run-1)]))
colnames(gradients) <- c("norm_alpha", "norm_beta", "norm_C", "norm_D")
gradients$iteration <- c(1:nrow(gradients))

grad_alpha <- ggplot(gradients, aes(x=iteration, y=norm_alpha)) +
  geom_line()+
  # geom_point()+
  labs(x ="Iteration", y = "Gradient norm of alpha")+
  theme_bw()

grad_beta <- ggplot(gradients, aes(x=iteration, y=norm_beta)) +
  geom_line()+
  # geom_point()+
  labs(x ="Iteration", y = "Gradient norm of beta")+
  theme_bw()

grad_C <- ggplot(gradients, aes(x=iteration, y=norm_C)) +
  geom_line()+
  # geom_point()+
  labs(x ="Iteration", y = "Gradient norm of C")+
  theme_bw()

grad_D <- ggplot(gradients, aes(x=iteration, y=norm_D)) +
  geom_line()+
  # geom_point()+
  labs(x ="Iteration", y = "Gradient norm of D")+
  theme_bw()

grid.arrange(grad_alpha, grad_beta, grad_C, grad_D, ncol = 2, nrow = 2)

df_llh <- pars10k_grad$objective[c(2:pars10k_grad$run)]
df_llh <- data.frame(cbind(llh = df_llh, iteration = c(1:length(df_llh))))

plot_llh <- ggplot(df_llh, aes(x=iteration, y=llh)) +
  geom_line()+
  # geom_point()+
  labs(x ="Iteration", y = "Penalized negative log likelihood")+
  theme_bw(base_size = 13)

ggsave(file="Figures/grad_alpha.eps", plot = grad_alpha)
ggsave(file="Figures/grad_beta.eps", plot = grad_beta)
ggsave(file="Figures/grad_C.eps", plot = grad_C)
ggsave(file="Figures/grad_D.eps", plot = grad_D)
ggsave(file="Figures/llh.eps", plot = plot_llh)
