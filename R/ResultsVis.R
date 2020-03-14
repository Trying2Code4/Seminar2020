# install.packages("ROCR")
library(openxlsx)
library(ROCR)
library(readr)

# Load all prediction columns, and a CLICK column
outputRes <- readRDS("outputRes.RDS") #This is local
CBF_RRprobs <- read_delim("CBF_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)
LMF_RRprobs <- read_delim("LMF_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)

pred <- outputRes$prediction[ ,c("prediction", "CLICK")]

# Use ROCR package
pred1 <- ROCR::prediction(pred$prediction, pred$CLICK)
pred2 <- ROCR::prediction(CBF_RRprobs, pred$CLICK)
pred3 <- ROCR::prediction(LMF_RRprobs, pred$CLICK)

perf1 <- ROCR::performance(pred1,"tpr","fpr")
perf2 <- ROCR::performance(pred2,"tpr","fpr")
perf3 <- ROCR::performance(pred3,"tpr","fpr")

# Put it all in one dataframe (<- PROBLEM!)
plot(perf1, lty=1)
plot(perf2, colorize=FALSE, lty=2, add=TRUE)
plot(perf3, colorize=FALSE, lty=3, add=TRUE)








