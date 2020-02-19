# install.packages("recosystem")
library(recosystem)
library(dplyr)
library(readr)
library(softImpute)
library(spam)
library(tidyverse)
library(tictoc)
library(bigmemory)
library(RcppArmadillo)

set.seed(123)
df <- read_delim("~/Google Drive/Seminar 2020/Data/Observations_Report.csv",
                 ";", escape_double = FALSE, trim_ws = TRUE)
df <- df[order(df$USERID, df$OFFERID),]
df$USERID_ind <- group_indices(df, USERID)
df$ORDERID_ind <- group_indices(df, OFFERID)
df <- df[order(df$MAILID, decreasing=TRUE),]
data <- df[!duplicated(df[c(1,3)]),c(5,6,4)]
saveRDS(data, "data.RDS")
df <- readRDS("~/Dropbox/Uni/Master_Econometrie/Blok_3/Seminar2020/R/data.RDS")
names(df) <- c("USERID", "ORDERID", "CLICK")
df <- as.matrix(df)

# #select first 1000 users for first subset--------
# subset <- data[data$USERID_ind %in% 1:1000,]
# df$USERID_ind <- group_indices(df, USERID)
# df$ORDERID_ind <- group_indices(df, OFFERID)
# #Save subset
# saveRDS(subset, "Subset.RDS")
# 
# #Load subset
# df <- readRDS("~/Dropbox/Uni/Master_Econometrie/Blok_3/Seminar2020/R/Subset.RDS")
# df$USERID <- group_indices(df, USERID_ind)
# df$ORDERID <- group_indices(df, ORDERID_ind)
# df <- select(df, 3:5)
# df <- as.matrix(select(df, "USERID","ORDERID","CLICK"))


#--------------
#Tools
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

#ALGORITHM
#initialization of parameters
factors <- 2
priorsdu <- 1
priorsdi <- 1
priorlambdau <- 1/priorsdu
priorlambdai <- 1/priorsdi
nu <- length(unique(df[,1]))
ni <- length(unique(df[,2]))

#Alpha and beta's initialized with a zero, C and D with normal priors
alpha <- rep(0, nu)
beta <- rep(0,ni)
C <- matrix(rnorm(nu*factors,0,priorsdu),nu,factors)
D <- matrix(rnorm(ni*factors,0,priorsdi),ni,factors)

#Center required parameters for identification
beta <- scale(beta, scale = FALSE)
C <- scale(C, scale = FALSE)
D <- scale(D, scale = FALSE)

#Define low rank representation of gamma0
tic()

low_rankC <- cbind(C,alpha,rep(1,nu))
low_rankD <- cbind(D,rep(1,ni),beta)

#Calculate gamma0
gamma0 <- low_rankC%*%t(low_rankD)

#Retrieve indices for y=1 and y=0 from the input data
y1 <- df[which(df[,3]==1),c("USERID","ORDERID")]
y0 <- df[which(df[,3]==0),c("USERID","ORDERID")]

#Calculate respective first derivatives for both y=1 and y=0
y1 <- cbind(y1, "deriv" = -4*derf1(gamma0[y1]))
y0 <- cbind(y0, "deriv" = -4*derf2(gamma0[y0]))

#Combine the results in one matrix
y01 <- rbind(y1,y0)

#Turn this matrix to sparse, notice that the dims had to be manually set (for missing items probably
sparse <- sparseMatrix(i=(y01[,1]), j=(y01[,2]), x = y01[,3], dims = c(nu, ni))

#Calculating the H matrix for alpha update
H_slr <- splr(sparse,low_rankC,low_rankD)

#Updating alpha and beta
newalpha <- as.matrix((1/ni)* H_slr %*% rep(1,ni))

#Subtract the rowmean from H for the update for beta
low_rankC <- cbind(C,(alpha-rowMeans(H_slr)),rep(1,nu))
H_slr_rowmean <- splr(sparse,low_rankC,low_rankD)
newbeta <- as.matrix((1/nu) * t(t(rep(1,nu))%*%H_slr_rowmean))

#Updating the C and D
#Remove row and column mean from H
low_rankD <- cbind(D,rep(1,ni),(beta-colMeans(H_slr_rowmean)))
H_slr_rowandcolmean <-splr(sparse,low_rankC,low_rankD)

#Retrieve C and D from the svd.als function
results <- svd.als(H_slr_rowandcolmean, rank.max = factors, lambda = priorlambdau)
toc()
