library(recosystem)
library(dplyr)
library(readr)
library(softImpute)
library(spam)
library(tidyverse)
library(tictoc)
library(bigmemory)
library(RcppArmadillo)
library(Rcpp)
library(ggplot2)

sourceCpp("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/gammaui.cpp")
#sourceCpp("~/Dropbox/Uni/Master_Econometrie/Blok_3/Seminar2020/R/gammaui.cpp")


# Functions/tools ------------------------------------------------------------------------

derf1 <- function(x){
  return(-1 / (1 + exp(x)))
}

derf2 <- function(x){
  return(1 / (1 + exp(-x)))
}

mu <- function(x){
  return(1 / (1 + exp(-x)))
}

logllh1 <- function(x){
  return(log(1 + exp(-x)))
}

logllh0 <- function(x){
  return(x + log(1 + exp(-x)))
}


#' Create a train/test split, given a training set
#'
#' @param df df containing indices, click, and "ratio" columns (see "Preparing data")
#' @param onlyVar logical variable speciying whether rows/columns without variation (only)
#' @param cv indicates whether train test split is made for CV
#' @param ind vector with fold indices in case of CV
#' @param fold fold that should currently be the test set
#' zero's or one's are omitted
#'
#' @return returns a training set and the test set
trainTest <- function(df, onlyVar, cv=FALSE, ind=NULL, fold=NULL){
  # Formatting
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")
  
  # Make the test train split (test is 1)
  # In case of a random draw
  if(!cv){
    df$train_test <- rbinom(n = nrow(df), size = 1, prob = 0.2)
  }
  
  # In case of cross validation (recode to zeros and one's for ease)
  if (cv){
    df$train_test <- 0
    df$train_test[ind == fold] <- 1
  }
  
  df$prediction <- NA
  
  # Deleting the rows/columns without variation
  if (onlyVar) {
    
    # Split them (temporarily)
    df_test <- df[as.logical(df$train_test), ]
    df_train <- df[!(as.logical(df$train_test)), ]
    
    # Assign the 0 or 1 to test set obs where a ratio is 0 or 1 (prediction in advance)
    df_test$prediction[(df_test$ratioU == 0 | df_test$ratioO == 0)] <- 0
    df_test$prediction[(df_test$ratioU == 1 | df_test$ratioO == 1)] <- 1
    
    # Drop the train obs where a ratio is 0 or 1
    df_train <- df_train[!(df_train$ratioU == 0 | df_train$ratioO == 0 | 
                             df_train$ratioU == 1 | df_train$ratioO == 1), ]
    
    # Merge the two to make indices
    df <- dplyr::bind_rows(df_train, df_test)
  }
  
  # Create new indices. Make sure test is at bottom
  df <- df[order(df$train_test, df$USERID_ind, df$OFFERID_ind), ]
  df <- df %>% 
    mutate(USERID_indN = group_indices(., factor(USERID_ind, levels = unique(USERID_ind))))
  df <- df %>% 
    mutate(OFFERID_indN = group_indices(., factor(OFFERID_ind, levels = unique(OFFERID_ind))))
  
  # Split sets
  df_test <- df[as.logical(df$train_test), ]
  df_train <- df[!(as.logical(df$train_test)), c("USERID_indN", "OFFERID_indN", "CLICK", 
                                                 "ratioU", "ratioO")]
  
  #4. Return
  output <- list("df_train" = df_train, "df_test" = df_test)
  return(output)
}


#' Gives initial estimates for alpha, beta, C and D
#'
#' @param df training df containing ONLY indices and click
#' @param factors depth of C and D
#' @param priorsdu variance of normal distr for C
#' @param priorsdi variance of normal distr for D
#' @param initType method used for initialization (integer value)
#'
#' @return
#' @export
#'
#' @examples
initChoose <- function(df, factors, priorsdu, priorsdi, initType){
  # Formatting
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  nu <- max(df[ ,"USERID_ind"])
  ni <- max(df[ ,"OFFERID_ind"])
  
  # 0 for alpha and beta. Normal for C and D with mean 0
  if (initType == 1){ 
    #Alpha and beta's initialized with a zero, C and D with normal priors
    alpha <- rep(0, nu)
    beta <- rep(0, ni)
    C <- matrix(rnorm(nu * factors, 0, priorsdu), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, priorsdi), ni, factors)
  }
  # Old method by Colin that doesn't really work
  else if (initType == 2) {
    avg <- mean(df$CLICK)
    gammaAvg <- -1 * log(1/avg - 1)
    
    gamma <- matrix(rnorm(nu * ni, 0, 1), nu, ni) + gammaAvg
    svdGamma <- svd(gamma, nu = factors, nv = factors)
    
    Ctemp <- svdGamma$u %*% diag(sqrt(svdGamma$d))
    Dtemp <- svdGamma$v %*% diag(sqrt(svdGamma$d))
    
    alpha <- rowMeans(Ctemp)
    beta  <- rowMenas(Dtemp)
    
    C <- Ctemp - matrix(rep(alpha, factors), nu, factors)
    D <- Dtemp - matrix(rep(beta, factors), ni, factors)
    # Colin's method
  } else if (initType == 3){
    avg <- mean(df$CLICK)
    gammaAvg <- -1 * log(1/avg - 1)
    
    mean <- sqrt(1/factors * abs(gammaAvg))
    
    Ctemp <- matrix(rnorm(nu * factors, mean, priorsdu), nu, factors)
    Dtemp <- matrix(rnorm(ni * factors, -mean, priorsdu), ni, factors)
    
    alpha <- rowMeans(Ctemp)
    beta  <- rowMeans(Dtemp)
    C <- Ctemp - matrix(rep(alpha, factors), nu, factors)
    D <- Dtemp - matrix(rep(beta, factors), ni, factors)
    # Take alpha's that create avg click rate per user
  } else if (initType == 4){
    
    # Make user click averages
    temp <- df %>%
      group_by(USERID_ind) %>%
      summarize(meanCLICK = mean(CLICK)) %>%
      select(meanCLICK)
    
    # Give some value when this is 0 or 1 (otherwise gamma -> inf)
    temp[temp == 0] <- 0.01 # THINK ABOUT THIS
    temp[temp == 1] <- 0.99 
    
    # Calculate the gamma's that produce these click rates
    alpha <- as.matrix(-1 * log(1/temp - 1))
    
    # Simple zero means for the other parameters
    beta <- rep(0, ni)
    C <- matrix(rnorm(nu * factors, 0, priorsdu), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, priorsdi), ni, factors)
  }
  # Method 5, but adding columns means also
  else if (initType == 5){
    # Thijs' method
    nu <- max(df[ ,1])
    ni <- max(df[ ,2])
    
    # Make user click averages
    temp1 <- df %>%
      group_by(USERID_ind) %>%
      summarize(meanCLICK = mean(CLICK)) %>%
      select(meanCLICK)
    
    # Give some value when this is 0 or 1 (otherwise gamma -> inf)
    temp1[temp1 == 0] <- 0.01 # THINK ABOUT THIS
    temp1[temp1 == 1] <- 0.99 
    
    # Calculate the gamma's that produce these click rates
    alpha <- as.matrix(-1 * log(1/temp1 - 1))
    
    # Make beta
    df <- as.data.frame(t(scale(t(df), scale = FALSE)))
    
    temp2 <- df %>%
      group_by(OFFERID_ind) %>%
      summarize(meanCLICK = mean(CLICK)) %>%
      select(meanCLICK)
    
    temp2[temp2 == 0] <- 0.01 # THINK ABOUT THIS
    temp2[temp2 == 1] <- 0.99 
    
    beta <- as.matrix(-1 * log(1/temp2 - 1))
    
    C <- matrix(rnorm(nu * factors, 0, priorsdu), nu, factors)
    D <- matrix(rnorm(ni * factors, 0, priorsdi), ni, factors)
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
#' @param epsilon Convergence criteria
#'
#' @return returns parameters alpha, beta, C and D
parEst <- function(df, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, 
                   initType, llh, rmse, df_test=NULL, epsilon=NULL) {
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  # Initialization
  initPars <- initChoose(df, factors, priorsdu, priorsdi, initType)
  
  #Center required parameters for identification
  alpha <- initPars$alpha
  beta <- scale(initPars$beta, scale = FALSE)
  C <- scale(initPars$C, scale = FALSE)
  D <- scale(initPars$D, scale = FALSE)
  
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
  
  run <- 1
  
  if (!is.null(epsilon) || llh) {
    logllh_old <- sum(logllh1(gamma_y1)) + sum(logllh0(gamma_y0)) + 
      priorlambdau/2 * norm(C, type="F")^2 + priorlambdai/2 * norm(D, type="F")^2
  }
  
  if (llh) {
    # Keeping track of likelihoods
    logllh <- rep(NA, (iter+1))
    
    # Calculate log likelihood
    logllh[run] <- logllh_old
  } else{
    logllh <- NA
  }
  
  if (rmse){
    # Keeping track of rmse
    rmse_it <- rep(NA, (iter+1))
    temp <- getPredict(df_test, alpha, beta, C, D)
    predictions <- temp$prediction
    predictions[is.na(predictions)] <- 0
    actuals <- temp$CLICK
    
    rmse[run] <- sqrt(mean((predictions - actuals)^2))
  } else{
    rmse_it <- NA
  }
  
  while (run <= iter) {
    tic(paste("Complete iteration", run, sep = " "))
    #Define low rank representation of gamma0
    low_rankC <- cbind(C, alpha, rep(1, nu))
    low_rankD <- cbind(D, rep(1,ni), beta)
    
    #Calculate gamma0
    # gamma0 <- low_rankC %*% t(low_rankD)
    
    #Calculate respective first derivatives for both y=1 and y=0
    df1[,"deriv"] <- -4 * derf1(gamma_y1)
    df0[,"deriv"] <- -4 * derf2(gamma_y0)
    
    # df1 <- cbind(y1, "deriv" = -4 * derf1(gamma0[y1]))
    # df0 <- cbind(y0, "deriv" = -4 * derf2(gamma0[y0]))
    
    #Combine the results in one matrix
    df01 <- rbind(df1, df0)
    
    #Turn this matrix to sparse, notice that the dims had to be manually set
    # (for missing items probably)
    sparse <- sparseMatrix(i = (df01[ ,"USERID_ind"]), j = (df01[ ,"OFFERID_ind"]),
                           x = df01[ ,"deriv"], dims = c(nu, ni))
    
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
    
    # Updating gamma
    gamma_y1 <- get_gamma0(y1[,1], y1[,2], alpha, beta, C, D)
    gamma_y0 <- get_gamma0(y0[,1], y0[,2], alpha, beta, C, D)
    
    if (run>2 && !is.null(epsilon)) {
      logllh_old <- logllh_new
    }
    
    if (!is.null(epsilon) || llh) {
      logllh_new <- sum(logllh1(gamma_y1)) + sum(logllh0(gamma_y0)) +
        priorlambdau / 2 * norm(C, type = "F") ^ 2 + priorlambdai / 2 * norm(D, type = "F") ^ 2
    }
    if (llh){
      # Log Likelihood of current iteration
      logllh[run] <- logllh_new
    }
    
    if (rmse){
      # RMSE of current iteration
      temp <- getPredict(df_test, alpha, beta, C, D)
      predictions <- temp$prediction
      predictions[is.na(predictions)] <- 0
      actuals <- temp$CLICK
      
      rmse_it[run] <- sqrt(mean((predictions - actuals)^2))
    }
   toc()
    
    if (!is.null(epsilon)) {
      if (abs((logllh_new-logllh_old)/logllh_old) < epsilon) break
    }
  }
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "logllh" = logllh, 
                 "rmse" = rmse_it, "run" = run)
  return(output)
}

#' Get predictions for a test set
#'
#' @param df Test set consisting of user and offer id's and CLICK (NA or value)
#' @param alpha parameter estimate alpha
#' @param beta parameter estimate beta
#' @param C parameter estimate C
#' @param D parameter estimate D
#' @param uniqueU basically a vector used as a dictionary
#' @param uniqueI basically a vector used as a dictionary
#'
#' @return dataframe including predictions, NA for unknown user/item
getPredict <- function(df, alpha, beta, C, D){
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK", "prediction")
  
  # By using the size of C and D, we can infer which obs are missing in training
  maxU <- nrow(C)
  maxI <- nrow(D)
  
  # Marking offer/items that are non existent in the training set
  df$nonMiss <- ((df[ ,"USERID_ind"]  <= maxU) & (df[ ,"OFFERID_ind"] <= maxI))
  
  # Predciting for the non missing obs
  # Get the non missing indices
  nonMiss <- as.matrix(df[df$nonMiss, c("USERID_ind", "OFFERID_ind")])
  
  # Calculating gamma
  gamma <- get_gamma0(nonMiss[,1], nonMiss[,2], alpha, beta, C, D)
  
  # And predict useing the llh function and gamma
  df$prediction[df$nonMiss] <- mu(gamma)
  
  return(df)
}

#' Run the full algorithm
#'
#' @param df_train
#' @param df_test 
#' @param factors 
#' @param priorsdu 
#' @param priorsdi 
#' @param priorlambdau 
#' @param priorlambdai 
#' @param iter 
#' @param initType 
#' @param onlyVar 
#'
#' @return
#' @export
#'
#' @examples
fullAlg <- function(df_train, df_test, factors, priorsdu, priorsdi, priorlambdau, 
                    priorlambdai, iter, initType, llh=FALSE, rmse=FALSE, epsilon=NULL){
  # Estimating parameters
  tic("2. Estimating parameters")
  pars <- parEst(df_train, factors, priorsdu, priorsdi, priorlambdau, priorlambdai, iter, 
                 initType, llh, rmse, df_test, epsilon)
  toc()
  
  # Getting predictions
  tic("3. Getting predictions")
  results <- getPredict(df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK", "prediction")], 
                        pars$alpha, pars$beta, pars$C, pars$D)
  toc()
  
  # RMSE
  # What to do with the NANs (some majority rule)
  results$prediction[is.na(results$prediction)] <- 0
  RMSE <- sqrt(mean((results$prediction - results$CLICK)^2))
  
  # Calculate confusion matrix
  tresh <- 0.02192184 # average click rate
  results$predictionBin <- rep(0, length(results$prediction))
  results$predictionBin[results$prediction > tresh] <- 1
  
  # True positives:
  TP <- sum(results$predictionBin == 1 & results$CLICK == 1)
  TN <- sum(results$predictionBin == 0 & results$CLICK == 0)
  FP <- sum(results$predictionBin == 1 & results$CLICK == 0)
  FN <- sum(results$predictionBin == 0 & results$CLICK == 1)
  
  confusion <- list("TP" = TP, "TN" = TN, "FP" = FP, "FN" = FN)
  
  # Output
  output <- list("parameters" = pars, "prediction" = results, "RMSE" = RMSE, 
                 "confusion" = confusion)
  return(output)
}

crossValidate <- function(df, FACTORS, PRIORS, INITTYPE, ONLYVAR, folds, iter, epsilon){
  
  # Initialize a multidimensional output array
  # Rows are all the possible permutations of the huperparameters * folds
  rows <- (length(ONLYVAR) * length(FACTORS) * length(PRIORS) * length(INITTYPE)) * folds
  
  # Columns for the hyperparameters, plus a name variable, and then all the results you want
  # these are: rmse, TP (true positive(1)), TN, FP, FN, number of iterations
  columns <- 4 + 1 + 6
  
  # Initialize the df (depth is the number of folds)
  CVoutput <- data.frame(matrix(NA, nrow = rows, ncol = columns))
  names(CVoutput) <- c("Factor", "PriorS", "initType", "onlyVar", "Specification",
                       "RMSE", "TP", "TN", "FP", "FN", "iter")
  
  # Now we loop
  # First we make the folds
  # Randomly shuffle the data
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
      set.seed(123)
      split <- trainTest(df, onlyVar, cv = TRUE, ind = foldInd, fold = z)
      df_train <-split$df_train[ ,c("USERID_indN", "OFFERID_indN", "CLICK")]
      df_test <- split$df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK", "prediction")]
      
      # Loop the other hyperparameters
      for (b in 1:length(FACTORS)){
        for (c in 1:length(PRIORS)){
          for (d in 1:length(INITTYPE)){
            tic(paste("Run", row, "out of", rows, "in fold", z, "out of ", folds))
            factors <- FACTORS[b]
            priorsdu <- PRIORS[c]
            priorsdi <- PRIORS[c]
            initType <- INITTYPE[d]
            priorlambdau <- 1/priorsdu
            priorlambdai <- 1/priorsdi
            
            # Run the algorithm
            invisible(
            output <- fullAlg(df_train, df_test, factors, priorsdu, priorsdi, priorlambdau, 
                              priorlambdai, iter, initType, epsilon)
            )
            # Fill the array with output
            CVoutput[row, 1] <- factors
            CVoutput[row, 2] <- priorsdu
            CVoutput[row, 3] <- initType
            CVoutput[row, 4] <- onlyVar
            
            # The name
            CVoutput[row, 5] <- paste("Factor = ", factors, "PriorS = ", priorsdu, 
                                         "initType = ", initType, "onlyVar" = onlyVar,
                                         sep = "")
            
            # Performance variables
            CVoutput[row, 6] <- output$RMSE
            CVoutput[row, 7] <- output$confusion$TP
            CVoutput[row, 8] <- output$confusion$TN
            CVoutput[row, 9] <- output$confusion$FP
            CVoutput[row, 10] <- output$confusion$TP
            CVoutput[row, 11] <- output$parameters$run - 1
            
            row <- row+1
            toc()
          }
        }
      }
    }
  }
  return(CVoutput)
}

baselinePred <- function(df_train, df_test){
  # initialize column with majority
  df_test$predUser <- 0
  df_test$predOffer <- 0
  df_test$predOverall <- mean(df_train$CLICK)
  df_test$predMajority <- 0
  
  # Fill in predictions where available
  df_test$predUser <- df_test$ratioU[!is.na(df_test$ratioU)]
  df_test$predOffer <- df_test$ratioO[!is.na(df_test$ratioO)]
  
  rmseUser <- sqrt(mean((df_test$predUser - df_test$CLICK)^2))
  rmseOffer <- sqrt(mean((df_test$predOffer - df_test$CLICK)^2))
  rmseOverall <- sqrt(mean((df_test$predOverall - df_test$CLICK)^2))
  rmseMajority <- sqrt(mean((df_test$predMajority - df_test$CLICK)^2))
  
  output <- list("rmseUser" = rmseUser, "rmseOffer" = rmseOffer, 
                 "rmseOverall" = rmseOverall, "rmseMajority" = rmseMajority)
  
  return(output)
}

  


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
  unite("MailOffer", c("MAILID" ,"OFFERID"), sep = "_")%>%
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
df <- readRDS("~/Google Drive/Seminar 2020/Data/df_train")
df <- df[ , c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")]

# Setting parameters
factors <- 2
priorsdu <- 1
priorsdi <- 1
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi
iter <- 100
initType <- 4
onlyVar <- FALSE
epsilon <- 

tic("1. Train test split")
split <- trainTest(df, onlyVar)
df_train <-split$df_train[ ,c("USERID_indN", "OFFERID_indN", "CLICK")]
df_test <- split$df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK", "prediction")]
rm("split")
toc()

output <- fullAlg(df_train, df_test, factors, priorsdu, priorsdi, priorlambdau, 
                  priorlambdai, iter, initType, llh = TRUE, rmse = TRUE)

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
priorsdu <- 1
priorsdi <- 1
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi
iter <- 200
initType <- 4
onlyVar <- TRUE
llh <- TRUE
rmse <- TRUE
epsilon <- 0.001

set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_indN", "OFFERID_indN", "CLICK")]
df_test <- split$df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK", "prediction")]

df_train2 <- split$df_train[ ,c("USERID_indN", "OFFERID_indN", "CLICK")]
df_test2 <- split$df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK", "ratioU", "ratioO")]

rm("split")

output <- fullAlg(df_train, df_test, factors, priorsdu, priorsdi, priorlambdau, 
                  priorlambdai, iter, initType, llh, rmse, epsilon)

baseline <- baselinePred(df_train2, df_test2)


# Visualization
hist(output2$prediction$prediction)
xdata <- seq(1, iter+1)
plot(xdata, output2$parameters$logllh, col="blue")
plot(xdata, output2$parameters$rmse_it, col="red")

# Cross validation -----------------------------------------------------------------------
# Import train set
# Make sure the names are correct
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- df[df$USERID_ind < 10000, c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")]


# Input whichever hyperparameters you want to test
FACTORS <- c(2, 4)
PRIORS <- c(1, 10)
INITTYPE <- c(4)
ONLYVAR <- c(TRUE)
folds <- 2
iter <- 2
epsilon <- 0.01

CVoutput <- crossValidate(df, FACTORS, PRIORS, INITTYPE, ONLYVAR, folds, iter, epsilon)

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

test <- getPredict(df_test[ ,c("USERID_indN", "OFFERID_indN", "CLICK", "prediction")], 
                   alpha, beta, C, D)

test$prediction[is.na(test$prediction)] <- 0

hist(test$prediction)

# Problem with indices ------------------------------------------------------------------



