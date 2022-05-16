rm(list=ls())     # Clean memory
cat("\014")       # Clear Console
graphics.off()    # Close graphs
library(moments)
library(tictoc)

# Load function library
ScriptLocation <- paste0(dirname(rstudioapi::getSourceEditorContext()$path),"/functions/GARCH11Functions.R")
source(ScriptLocation)

#  Simulation parameters for GARCH(1,1) Monte Carlo
theta <- matrix(c(0.1, 0.05, 0.8), nrow=3, ncol=1)
T <- 1E4
Nsim <- 1E3
set.seed(123)

# Initialise matrices
ParaEst <- matrix(NA, nrow=3, ncol=Nsim)
tstat <- matrix(NA, nrow=3, ncol=Nsim)

# Estimate GARCH(1,1) data series
tic()
for (simiter in 1:Nsim)
{
    # Report progress
    if (simiter%%100==0) cat("Iteration", simiter, "out of", Nsim,"\n")
  
    # Generate GARCH(1,1) data set
    data <- GenerateGARCH11(theta, T)
    
    # Estimation
    est <- EstimateGARCH11(data, thetastart = theta)
    ParaEst[, simiter] <- est[[1]]
    tstat[, simiter] <- sqrt(T)*(est[[1]]-theta)/sqrt( diag(est[[3]]) ) # Complex sqrt automatically converted to NaN
}
toc()

SingleMonteCarloRun <- function(theta, T)
{
    # Generate GARCH(1,1) data set
    data <- GenerateGARCH11(theta, T)
    
    # Estimation
    est <- EstimateGARCH11(data, thetastart = theta)
    tstat <- sqrt(T)*(est[[1]]-theta)/sqrt( diag(est[[3]]) ) # Complex sqrt automatically converted to NaN
    return(tstat)
}


tic()
# Implementation using apply (no speed gains to find)
randomSamples <- sapply(c(1:Nsim), function(x) SingleMonteCarloRun(theta, T) )
toc()

# Omit outliers in histogram
SubHistogram <- function(datalist, lowerbound, upperbound)
{
  data2plot <- datalist[lowerbound<datalist & datalist<upperbound]
  hist(data2plot, breaks=20, freq=FALSE, xlim = c(lowerbound, upperbound))
}
  
SubHistogram(tstat[1,], -5, 5)
SubHistogram(tstat[2,], -5, 5)
SubHistogram(tstat[3,], -5, 5)