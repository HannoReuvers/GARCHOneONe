====== README =====

GARCH(1,1) generation and estimation code. Implementations in Matlab, Python, and R are available.

===== FUNCTIONS =====

The code for the three programming languages is available in separate folders. Within these folders you will find a functions directory containing:
 - EstimateGARCH11 to conduct inference on a GARCH(1,1) process
 - GARCH11ObjFunc2Minimise computes the conditional Gaussian quasi-likelihood
 - GenerateGARCH11 generates a GARCH(1,1) time series with N(0,1) innovations

===== OTHER FILES ====

 - AEX.csv contains data from the Amsterdam exchange index
 - GarchAEXEstimation.m (only in the Matlab folder) replicates the empirical results reported at https://hannoreuvers.github.io/post/garch11/
