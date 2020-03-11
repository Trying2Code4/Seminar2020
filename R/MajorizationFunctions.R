library(softImpute)
library(tidyverse)
library(tictoc)
library(openxlsx)

# FUNCTIONS FOR MAJORIZATION ------------------------------------------------------------------------

#' Partial loglikelihood when observed value is 1
#'
#' @param x vector of doubles
#'
#' @return Partial loglikelihood when observed value is 1
#' 
logllh1 <- function(x){
  return(log(1 + exp(-x)))
}

#' Partial loglikelihood when observed value is 0
#'
#' @param x vector of doubles
#'
#' @return Partial loglikelihood when observed value is 0
#'
logllh0 <- function(x){
  return(x + log(1 + exp(-x)))
}

#' Derivative of logllh1
#'
#' @param x vector of doubles
#'
#' @return Derivative of logllh1, evaluated in x
#'
derf1 <- function(x){
  return(-1 / (1 + exp(x)))
}

#' Derivative of logllh0
#'
#' @param x vector of doubles
#'
#' @return Derivative of logllh0, evaluated in x
#'
derf2 <- function(x){
  return(1 / (1 + exp(-x)))
}

#' Logistic function
#'
#' @param x vector of doubles
#'
#' @return Logistic function evaluated in x
#'
mu <- function(x){
  return(1 / (1 + exp(-x)))
}

#' Create a train/test split, given a training set
#'
#' @param df df containing indices and clicks
#' @param onlyVar boolean specifying whether rows without variation should be removed
#' @param cv indicates whether train test split is made for CV
#' @param ind vector with fold indices in case of CV
#' @param fold fold that should currently be the test set
#' @param test_size fraction of data that should be included in the test split, ignored if cv != NULL
#'
#' @return training set, corresponding test set, and global mean click rate of training set
#' 
trainTest <- function(df, onlyVar, cv=FALSE, ind=NULL, fold=NULL, test_size = 0.2){
  # Formatting
  original <- names(df)
  names(df)[1:3] <- c("USERID", "OFFERID", "CLICK")
  
  # Make the test train split (test set has value 1)
  if (cv) {
    # In case of cross validation (folds are known beforehand)
    df$train_test <- 0
    df$train_test[ind == fold] <- 1
  }
  else {
    df$train_test <- rbinom(n = nrow(df), size = 1, prob = test_size)
  }
  
  names(df) <- c(original, "train_test")
  return(prepData(df[!as.logical(df$train_test), ], df[as.logical(df$train_test), ], onlyVar))
}

#' Gives initial estimates for alpha, beta, C and D
#'
#' @param df training df containing ONLY indices and click
#' @param factors number of latent factors
#' @param lambda penalty term
#' @param initType method used for initialization (integer value)
#' @param a_in initial alpha (if warm start)
#' @param b_in initial beta (if warm start)
#' @param C_in initial C (if warm start)
#' @param D_in initial D (if warm start)
#'
#' @return initial estimates of alpha, beta, C and D
#'
initChoose <- function(df, factors, lambda, initType, a_in = NULL, b_in = NULL,
                       C_in = NULL, D_in = NULL){
  # Formatting
  names(df)[1:3] <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  nu <- max(df[, "USERID_ind"])
  ni <- max(df[, "OFFERID_ind"])
  
  # 0 for alpha and beta. Normal for C and D with mean 0
  if (initType == 1){ 
    #Alpha and betas initialized with a zero, C and D with normal priors
    alpha <- rep(0, nu)
    beta <- rep(0, ni)
    C <- matrix(rnorm(nu * factors, 0, 1/lambda), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, 1/lambda), ni, factors)
    
    # Take alphas that create avg click rate per user
  } else if (initType == 2){
    
    # Make user click averages
    user_avg <- df %>%
      group_by(USERID_ind) %>%
      summarize(meanCLICK = mean(CLICK)) %>%
      select(meanCLICK)
    
    # Give some value when this is 0 or 1 (otherwise gamma -> inf)
    user_avg[user_avg == 0] <- 0.00001 # THINK ABOUT THIS
    user_avg[user_avg == 1] <- 0.99999
    
    # Calculate the gammas that produce these click rates
    alpha <- as.matrix(-1 * log(1/user_avg - 1))
    
    #Make offer click averages
    offer_avg <- df %>%
      group_by(OFFERID_ind) %>%
      summarize(meanCLICK = mean(CLICK)) %>%
      select(meanCLICK)
    
    # Give some value when this is 0 or 1 (otherwise gamma -> inf)
    offer_avg[offer_avg == 0] <- 0.00001 # THINK ABOUT THIS
    offer_avg[offer_avg == 1] <- 0.99999
    
    # Calculate the gammas that produce these click rates
    beta <- as.matrix(-1 * log(1/offer_avg - 1))
    
    C <- matrix(rnorm(nu * factors, 0, 1/lambda), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, 1/lambda), ni, factors)
  }
  
  #Center required parameters for identification
  beta <- scale(beta, scale = FALSE)
  C <- scale(C, scale = FALSE)
  D <- scale(D, scale = FALSE)
  
  # If a specific input for alpha, beta C or D is given,
  # overwrite previously defined parameters
  if (!is.null(a_in)){
    alpha <- a_in
  }
  if (!is.null(b_in)){
    beta <- b_in
  }
  if (!is.null(C_in)){
    C <- C_in
  }
  if (!is.null(D_in)){
    D <- D_in
  }
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D)
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
parEst <- function(df, factors, lambda, iter, initType, llh, rmse, df_test=NULL, 
                   epsilon=NULL, a_in = NULL, b_in = NULL, C_in = NULL, D_in = NULL) {
  names(df)[1:3] <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  # Initialization (including centering)
  initPars <- initChoose(df, factors, lambda, initType, a_in, b_in, C_in, D_in)
  
  alpha <- initPars$alpha
  beta <- initPars$beta
  C <- initPars$C
  D <- initPars$D
  
  df <- as.matrix(df)
  
  nu <- max(df[, "USERID_ind"])
  ni <- max(df[, "OFFERID_ind"])
  
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
  # alpha_track <- alpha
  # beta_track <- beta
  # C_track <- C
  # D_track <- D
  
  while (run <= iter) {
    # Keeping track of parameters
    # alpha_track <- cbind(alpha_track, alpha)
    # beta_track <- cbind(beta_track, beta)
    # C_track <- cbind(C_track, C)
    # D_track <- cbind(D_track, D)
    
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
  
  if (!(llh) && !is.null(epsilon)) {
    deviance_all = deviance_new
    objective_all = objective_new
  }
  # Keeping track
  # par_track <- list("alpha_track" = alpha_track, "beta_track" = beta_track, 
  #                   "C_track" = C_track, "D_track" = D_track)
  
  # output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
  #                "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all,
  #                "par_track" = par_track)
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
                 "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all)
  return(output)
}

#' Get predictions for a test set
#'
#' @param df Test set consisting of user and offer ids and CLICK (NA or value)
#' @param alpha parameter estimate alpha
#' @param beta parameter estimate beta
#' @param C parameter estimate C
#' @param D parameter estimate D
#'
#' @return dataframe including predictions, NA for unknown users/items
getPredict <- function(df, alpha, beta, C, D){
  names(df)[1:6] <- c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO", "prediction")
  
  # By using the size of C and D, we can infer which obs are missing in training
  maxU <- nrow(C)
  maxI <- nrow(D)
  
  # Marking offer/items that are in the training set and do not have a prediction yet
  df$nonMiss <- (df[ ,"USERID_ind"]  <= maxU & df[ ,"OFFERID_ind"] <= maxI & is.na(df[ ,"prediction"]))
  
  # Predciting for the non missing obs
  # Get the non missing indices
  nonMiss <- as.matrix(df[df$nonMiss, c("USERID_ind", "OFFERID_ind")])
  
  # Calculating gamma
  gamma <- get_gamma0(nonMiss[,1], nonMiss[,2], alpha, beta, C, D)
  
  # And predict useing the llh function and gamma
  df$prediction[df$nonMiss] <- mu(gamma)
  
  return(df)
}


#' Run full algorithm
#'
#' @param df_train training set 
#' @param df_test test set
#' @param factors number of latent dimensions
#' @param lambda penalty term
#' @param iter maximum number of iterations
#' @param initType method of initialisation
#' @param llh boolean for whether loglikelihood should be tracked
#' @param rmse boolean for whether rmse should be tracked
#' @param epsilon convergence criteria
#' @param a_in initial alpha (for warm start)
#' @param b_in initial beta (for warm start)
#' @param C_in initial C (for warm start)
#' @param D_in initial D (for warm start)
#'
#' @return parameters, predictions, and performance measures
#'
fullAlg <- function(df_train, df_test, factors, lambda, iter, initType, llh=FALSE, 
                    rmse=FALSE, epsilon=NULL, a_in = NULL, b_in = NULL, C_in = NULL, D_in = NULL,
                    globalMean){
  # Estimating parameters
  tic("2. Estimating parameters")
  pars <- parEst(df_train, factors, lambda, iter, initType, llh, rmse, df_test, 
                 epsilon, a_in, b_in, C_in, D_in)
  toc()
  
  # Getting predictions
  tic("3. Getting predictions")
  results <- getPredict(df_test[ ,c("USERID_ind", "OFFERID_ind", "CLICK",
                                    "ratioU", "ratioO", "prediction", "USERID", "OFFERID")], 
                        pars$alpha, pars$beta, pars$C, pars$D)
  toc()
  
  # RMSE
  # What to do with the NANs (global mean)
  results$prediction[is.na(results$prediction)] <- globalMean
  RMSE <- sqrt(mean((results$prediction - results$CLICK)^2))
  
  # Calculate confusion matrix
  threshold <- 0.02192184 # average click rate
  results$predictionBin <- rep(0, length(results$prediction))
  results$predictionBin[results$prediction > threshold] <- 1
  
  # True positives:
  TP <- sum(results$predictionBin == 1 & results$CLICK == 1)
  TN <- sum(results$predictionBin == 0 & results$CLICK == 0)
  FP <- sum(results$predictionBin == 1 & results$CLICK == 0)
  FN <- sum(results$predictionBin == 0 & results$CLICK == 1)
  
  confusion <- list("TP" = TP, "TN" = TN, "FP" = FP, "FN" = FN)
  
  # Output
  output <- list("parameters" = pars,
                 "prediction" = results[, c("USERID", "OFFERID","USERID_ind", "OFFERID_ind",
                                            "CLICK", "prediction", "predictionBin", "ratioU",
                                            "ratioO")], 
                 "RMSE" = RMSE,
                 "confusion" = confusion)
  return(output)
}


#' Function for cross-validating hyperparameters
#'
#' @param df dataframe
#' @param FACTORS vector of factors (number of latent dimensions) to use
#' @param LAMBDA vector of penalty terms to use
#' @param INITTYPE vector of methods for initiation to use
#' @param ONLYVAR whether or not observations without variation should be removed (can take TRUE, FALSE, or c(TRUE, FALSE))
#' @param folds number of folds
#' @param iter maximum number of iterations
#' @param epsilon convergence limit
#' @param warm boolean for whether or not to use warm start
#'
#' @return performance per combination of hyperparameters
#'
crossValidate <- function(df, FACTORS, LAMBDA, INITTYPE, ONLYVAR, folds, iter, 
                          epsilon, warm, file=NULL){
  
  # Initialize a multidimensional output array
  # Rows are all the possible permutations of the huperparameters * folds
  rows <- (length(ONLYVAR) * length(FACTORS) * length(LAMBDA) * length(INITTYPE)) * folds
  
  # Columns for the hyperparameters, plus a name variable, and then all the results you want
  # these are: rmse, TP (true positive(1)), TN, FP, FN, number of iterations, best baseline, epsilon
  columns <- 15
  
  # Initialize the df (depth is the number of folds)
  CVoutput <- data.frame(matrix(NA, nrow = rows, ncol = columns))
  names(CVoutput) <- c("Factor", "Lambda", "InitType", "OnlyVar", "Epsilon", "Specification",
                       "RetainedFactors", "RMSE", "TP", "TN", "FP", "FN", "Iter", "rmseUser", 
                       "DifferenceRMSE")
  
  # Now we loop
  # First we make the folds
  # Randomly shuffle the data
  set.seed(123)
  df <- df[sample(nrow(df)), ]
  
  # Then assign 1-5 fold indices
  foldInd <- cut(seq(1, nrow(df)), breaks = folds, labels = FALSE)
  
  row <- 1
  # Looping over your folds
  for (z in 1:folds){
    # Do onlyvar first because the train test split depends on it
    for (a in 1:length(ONLYVAR)){
      # Do onlyvar first because the train test split depends on it
      onlyVar <- ONLYVAR[a]
      
      # Make the train test split by using the foldInd and fold as input (see trainTest)
      split <- trainTest(df, onlyVar, cv = TRUE, ind = foldInd, fold = z)
      df_train <-split$df_train[ ,c("USERID_ind", "OFFERID_ind", "CLICK")]
      df_test <- split$df_test
      globalMean <- split$globalMean
      
      # Loop the other hyperparameters
      
      for (b in 1:length(FACTORS)){
        # Initialize the warm start objects (can only be used within a certain factor size)
        a_in <- NULL
        b_in <- NULL
        C_in <- NULL
        D_in <- NULL
        for (c in 1:length(LAMBDA)){
          for (d in 1:length(INITTYPE)){
            tic(paste("Run", row, "out of", rows, "in fold", z, "out of ", folds))
            factors <- FACTORS[b]
            lambda <- LAMBDA[c]
            initType <- INITTYPE[d]
            
            # Run the algorithm
            output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, 
                              epsilon = epsilon, a_in = a_in, b_in = b_in, C_in = C_in, D_in = D_in, 
                              globalMean = globalMean)
            
            # Fill the array with output
            CVoutput$Factor[row] <- factors
            CVoutput$Lambda[row] <- lambda
            CVoutput$InitType[row] <- initType
            CVoutput$OnlyVar[row] <- onlyVar
            CVoutput$Epsilon[row] <- epsilon
            
            # The name
            CVoutput$Specification[row] <- paste("Factor = ", factors, ", Lambda = ", lambda, 
                                                 ", initType = ", initType, ", onlyVar = ", onlyVar,
                                                 ", warm = ", warm, sep = "")
            
            # Performance variables
            CVoutput$RetainedFactors[row] <- output$parameters$factors[sum(!is.na(output$parameters$factors))]
            CVoutput$RMSE[row] <- output$RMSE
            CVoutput$TP[row] <- output$confusion$TP
            CVoutput$TN[row] <- output$confusion$TN
            CVoutput$FP[row] <- output$confusion$FP
            CVoutput$FN[row] <- output$confusion$FN
            CVoutput$Iter[row] <- output$parameters$run - 1
            CVoutput$rmseUser[row] <- baselinePred(df_test, globalMean)$rmseUser
            CVoutput$DifferenceRMSE[row] <- CVoutput$RMSE[row]-CVoutput$rmseUser[row]
            
            if (!is.null(file)) {
              write.xlsx(CVoutput, file=file, append = TRUE)
            }
            
            row <- row+1
            
            # In case of warm starts, keep track of the last parameters
            if (warm){
              a_in <- output$parameters$alpha
              b_in <- output$parameters$beta
              C_in <- output$parameters$C
              D_in <- output$parameters$D
            }
            
            toc()
            
            
          }
        }
      }
    }
  }
  
  # Create a mean table
  CVmean <- CVoutput %>% 
    group_by(Epsilon, Factor, Lambda, OnlyVar, InitType, Specification) %>% 
    summarise_all(mean) %>% 
    ungroup()
  
  return(list("CVoutput" = CVoutput, "CVmean" = CVmean))
}

#' Baseline predictions
#'
#' @param df_test test set
#' @param globalMean overall mean of the training data
#'
#' @return RMSE of baseline predictions
#' 
baselinePred <- function(df_test, globalMean){
  # initialize column with majority
  df_test$predUser <- globalMean
  df_test$predOffer <- globalMean
  df_test$predOverall <- globalMean
  df_test$predMajority <- globalMean
  
  # Fill in predictions where available
  df_test$predUser[!is.na(df_test$ratioU)] <- df_test$ratioU[!is.na(df_test$ratioU)]
  df_test$predOffer[!is.na(df_test$ratioO)] <- df_test$ratioO[!is.na(df_test$ratioO)]
  
  rmseUser <- sqrt(mean((df_test$predUser - df_test$CLICK)^2))
  rmseOffer <- sqrt(mean((df_test$predOffer - df_test$CLICK)^2))
  rmseOverall <- sqrt(mean((df_test$predOverall - df_test$CLICK)^2))
  rmseMajority <- sqrt(mean((df_test$predMajority - df_test$CLICK)^2))
  rmseComb <- sqrt(mean((0.5*df_test$predOffer + 0.5*df_test$predUser - df_test$CLICK)^2))
  
  output <- list("df_test" = df_test, "rmseUser" = rmseUser, "rmseOffer" = rmseOffer, 
                 "rmseOverall" = rmseOverall, "rmseMajority" = rmseMajority,
                 "rmseComb" = rmseComb)
  
  return(output)
}

prepData <- function(df_train, df_test, onlyVar){
  # Record the global mean
  globalMean <- mean(df_train$CLICK)
  
  # Prepare data for merging
  df_test$train_test <- 1
  df_train$train_test <- 0
  df <- rbind(df_train, df_test)
  
  # Use the train test function
  # User/offer click rate in train set (if user or offer not in train set, average is set at NaN)
  df <- df %>% group_by(USERID) %>% mutate(ratioU = sum((!as.logical(train_test))*CLICK)/sum(!as.logical(train_test))) %>% ungroup()
  df <- df %>% group_by(MailOffer) %>% mutate(ratioO = sum((!as.logical(train_test))*CLICK)/sum(!as.logical(train_test))) %>% ungroup()
  
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
    mutate(USERID_ind = group_indices(., factor(USERID, levels = unique(USERID))))
  df <- df %>% 
    mutate(OFFERID_ind = group_indices(., factor(MailOffer, levels = unique(MailOffer))))
  
  # Split sets
  df_test <- df[as.logical(df$train_test), c("USERID", "MailOffer", "USERID_ind", 
                                             "OFFERID_ind", "CLICK", "ratioU", "ratioO", 
                                             "prediction")]
  df_train <- df[!(as.logical(df$train_test)), c("USERID_ind", "OFFERID_ind", "CLICK", 
                                                 "ratioU", "ratioO")]
  
  output <- list("df_train" = df_train, "df_test" = df_test, "globalMean" = globalMean)
  return(output)
}