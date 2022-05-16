################################### FUNCTION: GenerateGARCH11 ###################################
#---INPUT VARIABLE(S)---
#   (1) theta: parameter vector in (omega, alpha, beta) format
#   (2) T: sample size
#---OUTPUT VARIABLE(S)---
#   (1) yt: simulated GARCH(1,1) time series
GenerateGARCH11 <- function(theta, T)
{
    # Select parameters
    omega <- theta[1]
    alpha <- theta[2]
    beta <- theta[3]
    
    #--- Simulate GARCH(1,1) process ---#
    # Initialize recursions from unconditional variance
    sigma2t0 <- omega/(1-alpha-beta)
    yt0 <- sqrt(sigma2t0)
    
    # Recursion to simulate GARCH(1,1) with Gaussian innovations
    sigma2t <- matrix(NaN, nrow=T, ncol=1)
    yt <- matrix(NaN, nrow=T, ncol=1)
    for (t in 1:T)
    {
        if ( t==1 ){
            sigma2t[t] <- omega + alpha*yt0^2 + beta*sigma2t0;
        } else {
            sigma2t[t] <- omega + alpha*yt[t-1]^2 + beta*sigma2t[t-1]
        }

        # Update data
        yt[t] <- sqrt(sigma2t[t])*rnorm(1)
    }
    return(yt)
}

################################### FUNCTION: GARCH11ObjFunc2Minimise ###################################
## DESCRIPTION: Objective function to MINIMISE for Gaussian QMLE
#---INPUT VARIABLE(S)---
#   (1) theta: parameter vector in (omega, alpha, beta) format
#   (2) yt: time series for inference
#---OUTPUT VARIABLE(S)---
#   (1) ScaledMinusLogLik: rescaled negative quasi log likelihood (without additive constants)
GARCH11ObjFunc2Minimise <- function(theta, yt)
{
    # Sample size
    T <- length(yt)
    
    # Select parameters
    omega <- theta[1]
    alpha <- theta[2]
    beta <- theta[3]
    
    # Estimate GARCH(1,1) process ---#
    # Initialize recursions from unconditional variance
    sigma2t0 <- max(omega/(1-alpha-beta), omega/(1-0.99)) # Use max to prevent negative sigma2t0
    yt0 <- sqrt(sigma2t0)
    
    # Recursion to reconstruct volatility process
    sigma2t <- matrix(NaN, nrow=T, ncol=1)
    for (t in 1:T){
        if (t==1){
            sigma2t[t] <- omega + alpha*yt0^2 + beta*sigma2t0
        } else {
            sigma2t[t] <- omega + alpha*yt[t-1]^2 + beta*sigma2t[t-1]
        }
    }
    
    ScaledMinusLogLik = colMeans( (yt^2)/sigma2t + log(sigma2t) )
    
    return(ScaledMinusLogLik)
}

################################### FUNCTION: EstimateGARCH11 ###################################
#---INPUT VARIABLE(S)---
#   (1) data: (Tx1) time series for GARCH estimation
#   (2) thetastart (OPTIONAL): starting trial for nonlinear optimizer
#---OUTPUT VARIABLE(S)---
#   (1) thetahat: estimate GARCH(1,1) parameters (omegahat, alphahat, betahat)
#   (2) etahat: estimated innovations
#   (3) AsymptCov: consistent estimator of the asymptotic covariance matrix
#   of the MLE
EstimateGARCH11 <- function(data, thetastart=matrix(c(0.03,0.05,0.94), nrow=3, ncol=1))
{
    ObjFunction <- function(theta){ GARCH11ObjFunc2Minimise(theta, data) }
    OptimResult <- optim(par=as.vector(thetastart), ObjFunction, method="L-BFGS-B", lower = c(0.01,0.01,0.01), hessian = TRUE)
    
    # Select OptimResult outputs
    thetahat <- OptimResult$par
    hessian <- OptimResult$hessian
    
    # Estimated innovations
    etahat <- InnovationFilter(data, thetahat)
    kurtosis_etahat <- kurtosis(etahat)
    
    # Asymptotic covariance matrix
    AsymptCov <- (kurtosis_etahat-1)*solve(hessian)
    
    return(list(thetahat, etahat, AsymptCov))
}

################################### FUNCTION: InnovationFilter ###################################
#---INPUT VARIABLE(S)---
#   (1) yt: (Tx1) time series for GARCH estimation
#   (2) thetahat: MLE of GARCH(1,1)
#---OUTPUT VARIABLE(S)---
#  (1) etahat: (Tx1) time series of estimated innovations
InnovationFilter <- function(yt, thetahat)
{
    # Sample size
    T <- length(yt)
    
    # Read quasi-MLEs from input
    omegahat <- thetahat[1]
    alphahat <- thetahat[2]
    betahat <- thetahat[3]
    
    # Recursion to reconstruct volatility process
    sigma20hat <- max(omegahat/(1-alphahat-betahat), omegahat/(1-0.99)) # Use max to prevent negative sigma2t0hat
    yt0hat <- sqrt(sigma20hat) 
    sigma2hat <- matrix(NA, nrow=T, ncol=1)
    for (t in 1:T)
    {
        if ( t==1 ){
            sigma2hat[t] <- omegahat + alphahat*yt0hat^2 + betahat*sigma20hat;
        } else {
            sigma2hat[t] <- omegahat + alphahat*yt[t-1]^2 + betahat*sigma2hat[t-1]
        }
    }
    
    # Estimated innovations
    etahat <- yt/sigma2hat
    
    return(etahat)
}