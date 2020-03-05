parEst <- function(df, factors, lambda, iter, initType, llh, rmse, df_test=NULL, 
                   epsilon=NULL, a_in = NULL, b_in = NULL, C_in = NULL, D_in = NULL) {
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  # Initialization
  initPars <- initChoose(df, factors, lambda, initType, a_in, b_in, C_in, 
                         D_in)
  
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
    
    # Keeping track of rmse FOR TRAIN!
    rmse_it2 <- rep(NA, (iter+1))
    pred <- getPredict2(df, alpha, beta, C, D) # USE TRAIN HERE!
    predictions <- pred$prediction
    # Majority rule
    predictions[is.na(predictions)] <- 0
    actuals <- pred$CLICK
    
    rmse_it2[run] <- sqrt(mean((predictions - actuals)^2))
    
  } else {
    rmse_it <- NA
  }
  
  # Keeping track of the number of factors
  factors_all <- rep(NA, (iter+1))
  factors_all[run] <- factors
  
  while (run <= iter) {
    # tic(paste("Complete iteration", run, sep = " "))
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
      
      # RMSE FOR TRAIN OF CURRENT ITERATION
      pred <- getPredict2(df, alpha, beta, C, D)
      predictions <- pred$prediction
      predictions[is.na(predictions)] <- 0
      actuals <- pred$CLICK
      
      rmse_it2[run] <- sqrt(mean((predictions - actuals)^2))
    }
    # toc()
    
    if (!is.null(epsilon)) {
      if (!is.infinite(objective_new) && !is.infinite(objective_old)) {
        print(paste("Iter", run, "Change in deviance is", (deviance_new-deviance_old)/deviance_old, sep=" "))
        print((objective_new-objective_old)/objective_old)
        if (abs((objective_new-objective_old)/objective_old) < epsilon) break
      }
    }
    
    # Keeping track of the number of factors
    if (factors == 1){
      factors_all[run] <- sum(results$d != 0)
    } else{
      factors_all[run] <- sum(diag(results$d) != 0)
    }
    
  }
  
  output <- list("alpha" = alpha, "beta" = beta, "C" = C, "D" = D, "objective" = objective_all, 
                 "deviance" = deviance_all, "rmse" = rmse_it, "run" = run, "factors" = factors_all,
                 "rmse2" = rmse_it2)
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
getPredict2 <- function(df, alpha, beta, C, D){
  #names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO", "prediction")
  df <- data.frame(df)
  names(df) <- c("USERID_ind", "OFFERID_ind", "CLICK")
  
  # By using the size of C and D, we can infer which obs are missing in training
  maxU <- nrow(C)
  maxI <- nrow(D)
  
  # Marking offer/items that are existent in the training set
  
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


### Running -----

# Use "Preparing data" first to get the df_train object
df <- readRDS("/Users/colinhuliselan/Documents/Master/Seminar/Code/SeminarR/df_train")
df <- df[, c("USERID_ind", "OFFERID_ind", "CLICK", "ratioU", "ratioO")]

# Run this if you want to use a subset of data!
df <- df[df$USERID_ind < 10000, ]

# Setting parameters
factors <- 4
lambda <- 1
iter <- 100
initType <- 2
onlyVar <- TRUE
llh <- TRUE
rmse <- TRUE
epsilon <- 1/10^5

set.seed(50)
split <- trainTest(df, onlyVar)
df_train <- split$df_train[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK")]
df_test <- split$df_test[ ,c("USERID_ind_new", "OFFERID_ind_new", "CLICK", "ratioU", "ratioO", "prediction")]
rm("split")

# OVERWRITE TO CREATE BULLSHIT CLICKS
# df_test$CLICK <- rbinom(n = nrow(df_test), size = 1, prob = 0.2)


output <- fullAlg(df_train, df_test, factors, lambda, iter, initType, llh, 
                  rmse, epsilon)



debug(getPredict2)
undebug(getPredict2)

baseline <- baselinePred(df_train, df_test)

# Visualization
plot(output$parameters$factors[1:sum(!is.na(output$parameters$factors))],
     col="orange", type = "o", ylab="Number of factors", xlab="Iteration")
plot(output$parameters$objective[1:sum(!is.na(output$parameters$objective))],
     col="blue", type = "l", ylab="Objective Function", xlab="Iteration")
plot(output$parameters$deviance[1:sum(!is.na(output$parameters$deviance))],
     col="green", type = "l", ylab="Deviance", xlab="Iteration")
plot(output$parameters$rmse[1:sum(!is.na(output$parameters$rmse))],
     col="red", type = "o", ylab="RMSE", xlab="Iteration")
lines(output$parameters$rmse2[1:sum(!is.na(output$parameters$rmse2))],
     col="purple", type = "o", ylab="RMSE2", xlab="Iteration")
