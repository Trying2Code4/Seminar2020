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

### Trying a simulated dataset
n <- 1000
p <- 200
f <- 3

# Underlying model
alpha <- runif(n, min=1, max=10)
beta <- runif(p, min=1, max=10)
C <- matrix(rnorm(n * f, 0, 5), n, f)
D <- matrix(rnorm(n * p, 0, 5), n, p)
gamma <- alpha + beta + C %*% t(D) + matrix(rnorm(n*p, 0, 1), n, p)
probability <- exp(gamma) / (1 + exp(gamma))

# Create a subset with a certain sparsity level
sparsity <- 0.1
USERID <- sample(1:n, sparsity*n*p)
OFFERID <- sample(1:p, sparsity*n*p)
df <- data.frame("USERID" = USERID, "OFFERID" = OFFERID)
