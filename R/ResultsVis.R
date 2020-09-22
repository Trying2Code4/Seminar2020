# install.packages("ROCR")
library(openxlsx)
library(ROCR)
library(readr)
library(xtable)

# Load all prediction columns, and a CLICK column
outputRes <- readRDS("outputRes.RDS") #This is local
CBF_RRprobs <- read_delim("CBF_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)
LMF_RRprobs <- read_delim("LMF_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)
Hybrids_RRprobs <- read_delim("Hybrids_RRprobs.csv", ",", escape_double = FALSE, trim_ws = TRUE)

pred <- outputRes$prediction[ ,c("prediction", "CLICK")]

# Use ROCR package
predLMF <- ROCR::prediction(LMF_RRprobs, pred$CLICK)
predMM <- ROCR::prediction(pred$prediction, pred$CLICK)
predCBF <- ROCR::prediction(CBF_RRprobs, pred$CLICK)
predH1 <- ROCR::prediction(Hybrids_RRprobs$H1, Hybrids_RRprobs$CLICK)
predH2 <- ROCR::prediction(Hybrids_RRprobs$H2, Hybrids_RRprobs$CLICK)

# ROC plot
perfLMF <- ROCR::performance(predLMF,"tpr","fpr")
perfMM <- ROCR::performance(predMM,"tpr","fpr")
perfCBF <- ROCR::performance(predCBF,"tpr","fpr")
perfH1 <- ROCR::performance(predH1,"tpr","fpr")
perfH2 <- ROCR::performance(predH2,"tpr","fpr")

plot(perfLMF, colorize=FALSE, lty=1, colour="#bfbdbd", downsampling=0.1)
plot(perfMM, colorize=FALSE, lty=2, colour="#bfbdbd", add=TRUE, downsampling=0.1)
plot(perfCBF, colorize=FALSE, lty=3, colour="#bfbdbd", add=TRUE, downsampling=0.1)
plot(perfH1, colorize=FALSE, lty=1, colour="black", add=TRUE, downsampling=0.1)
plot(perfH2, colorize=FALSE, lty=2, colour="black", add=TRUE, downsampling=0.1)

# Putting it all in one
predictions <- cbind(LMF_RRprobs, pred$prediction, CBF_RRprobs, Hybrids_RRprobs$H1, Hybrids_RRprobs$H2)
labels <- cbind(pred$CLICK, pred$CLICK, pred$CLICK, Hybrids_RRprobs$CLICK, Hybrids_RRprobs$CLICK)
predfull <- ROCR::prediction(predictions, labels)
perffull <- ROCR::performance(predfull, "tpr", "fpr")
# testplot <- plot(perffull, colorize=FALSE, downsampling=0.1)

# AUC full table
perffullAUC <- ROCR::performance(predfull, "auc")
AUCout <- as.data.frame(t(unlist(perffullAUC@y.values)))
colnames(AUCout) <- c("ISGA", "MM", "CBF", "H1", "H2")
rownames(AUCout) <- c("AUC")
print(xtable(AUCout, type = "latex", digits=rep(4,6)), 
      file = "/Users/colinhuliselan/Documents/Master/Seminar/Latex/AUCout.tex")


