
# Calculates the minus log likelihood
minusLogL <- function(Y, alpha, beta, C, D, lambda_u, lambda_i) {
  Gamma <- alpha%*%t(rep(1, ncol(Y))) + rep(1, nrow(Y))%*%t(beta) + C%*%t(D)
  f1 <- log(1+exp(-Gamma))
  sum <- sum(Y%*%f1 + (1-Y)%*%(Gamma+f1), na.rm=TRUE) + (lambda_u/2)*norm(C) + (lambda_i/2)*norm(D)
  return(sum)
}

# # Returns the value of f1
# f1 <- function(gamma) {
#   return(log(1+exp(-gamma)))
# }
# 
# # Returns the value of f2
# f2 <- function(gamma) {
#   return(gamma + f1(gamma))
# }
# 
# # Returns the value of f3 (which is always zero)
# # Included just for completeness' sake, can probably be removed later
# f3 <- function(gamma) {
#   return(0)
# }

# Returns the derivative of f1 for each element of Gamma
df1 <- function(Gamma) {
  return(-1/(1+exp(Gamma)))
}

# Returns the derivative of f2 for each element of Gamma
df2 <- function(Gamma) {
  return(1+df1(Gamma))
}

# Returns the value of g (the partial majorization function)
# Included just for completeness' sake, can probably be removed later
g <- function(a=(1/8), b, c, gamma) {
  return(a*gamma^2 - 2*b*gamma + c)
}

# Returns the value of a for a given value of gamma
# Included just for completeness' sake, can probably be removed later
a <- function(gamma0) {
  return(1/8)
}

# Returns the value of b for a given value of gamma
b <- function(gamma0, y=NA) {
  base <- (1/8)*gamma0
  if (y==1) {
    return(base - (1/2)*df1(gamma0))
  }
  else if (y==0) {
    return(base - (1/2)*df2(gamma0))
  }
  return(base)
}

# Returns the value of c for a given value of gamma
# Included just for completeness' sake, can probably be removed later
c <- function(gamma0, y=NA) {
  base <- (1/8)*gamma0^2
  base2 <- base - 2*b(gamma0, y)*gamma0
  if (y==1) {
    return(f1(gamma0) - base2)
  }
  else if (y==0) {
    return(f2(gamma0) - base2)
  }
  return(base)
}