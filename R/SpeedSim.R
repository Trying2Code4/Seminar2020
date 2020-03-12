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
          iter <- 5
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

#### Run it ------------------------------------------------------------------------------

debug(makeData)
# Create data
tic("simple model")
df <- makeData(nu=300000, ni=2000, sparsity=0.05, model=F)

# Or load data
df <- readRDS("df_train.RDS")

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
NU <- c(1, 1000, 10000, 100000, 1000000)
NI <- c(100)
SPARSITY <- c(0.05)
FACTORS <- c(5, 10, 15, 20)

output <- speedSim(NU, NI, SPARSITY, FACTORS, file="speedsim1.xlsx")

debug(speedSim)


