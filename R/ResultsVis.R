# install.packages("ROCR")
library(openxlsx)
library(ROCR)
library(readr)

# Load all prediction columns, and a CLICK column
outputRes <- readRDS("outputRes.RDS") #This is local
CBF_RRprobs <- read_delim("CBF_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)
LMF_RRprobs <- read_delim("LMF_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)
Hybrids_RRprobs <- read_delim("Hybrids_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)

pred <- outputRes$prediction[ ,c("prediction", "CLICK")]

# Use ROCR package
pred1 <- ROCR::prediction(pred$prediction, pred$CLICK)
pred2 <- ROCR::prediction(CBF_RRprobs, pred$CLICK)
pred3 <- ROCR::prediction(LMF_RRprobs, pred$CLICK)
pred4 <- ROCR::prediction(Hybrids_RRprobs$H1, Hybrids_RRprobs$CLICK)
pred5 <- ROCR::prediction(Hybrids_RRprobs$H2, Hybrids_RRprobs$CLICK)
test

perf1 <- ROCR::performance(pred1,"tpr","fpr")
perf2 <- ROCR::performance(pred2,"tpr","fpr")
perf3 <- ROCR::performance(pred3,"tpr","fpr")
perf4 <- ROCR::performance(pred4,"tpr","fpr")
perf5 <- ROCR::performance(pred5,"tpr","fpr")

# Put it all in one plot
plot(perf1, colorize=FALSE, lty=1, colour="#bfbdbd",)
plot(perf2, colorize=FALSE, lty=2, colour="#bfbdbd", add=TRUE)
plot(perf3, colorize=FALSE, lty=3, colour="#bfbdbd", add=TRUE)
plot(perf3, colorize=FALSE, lty=3, colour="black", add=TRUE)
plot(perf3, colorize=FALSE, lty=3, colour="black", add=TRUE)

# Getting the AUC
perf1AUC <- ROCR::performance(pred1, "auc")
perf2AUC <- ROCR::performance(pred2, "auc")
perf3AUC <- ROCR::performance(pred3, "auc")
perf4AUC <- ROCR::performance(pred4, "auc")
perf5AUC <- ROCR::performance(pred5, "auc")

# Putting it all in one
predictions <- cbind(pred$prediction, CBF_RRprobs, LMF_RRprobs, Hybrids_RRprobs$H1, Hybrids_RRprobs$H2)
labels <- cbind(pred$CLICK, pred$CLICK, pred$CLICK, Hybrids_RRprobs$CLICK, Hybrids_RRprobs$CLICK)
predfull <- ROCR::prediction(predictions, labels)
perffull <- ROCR::performance(predfull, "tpr", "fpr")
plot(perffull, colorize=FALSE, lty=1, colour="#bfbdbd",)



