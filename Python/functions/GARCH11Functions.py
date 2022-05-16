################################### FUNCTION: GenerateGARCH11 ###################################
def GenerateGARCH11(theta, T):
    """
    DESCRIPTION: Simulate GARCH(1,1) process
    --- INPUT VARIABLE(S) ---
    (1) theta: parameter vector in (omega, alpha, beta) formate
    (2) T: sample size
    --- OUTPUT VARIABLE(S) ---
    (1) yt: simulated GARCH(1,1) time series
    """
    # Select parameters
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]

    #--- Simulate GARCH(1,1) process ---#
    # Initialize recursions from unconditional variance
    sigma2t0 = omega/(1-alpha-beta)
    yt0 = np.sqrt(sigma2t0)

    # Recursion to simulate GARCH(1,1) with Gaussian innovations
    sigma2t = np.empty( (T, 1) ); sigma2t[:]= np.NaN
    yt = np.empty( (T, 1) ); yt[:] = np.NaN
    for t in range(0, T):
        if t == 0:
            sigma2t[t] = omega + alpha*yt0**2 + beta*sigma2t0
        else:
            sigma2t[t] = omega + alpha*yt[t-1]**2 + beta*sigma2t[t-1]

        # Update data
        yt[t] = np.sqrt(sigma2t[t])*np.random.normal()

    # Return output
    return yt

################################### FUNCTION: GARCH11ObjFunc2Minimise ###################################
def GARCH11ObjFunc2Minimise(theta, yt):
    """
    DESCRIPTION: Objective function to MINIMISE for Gaussian QMLE
    --- INPUT VARIABLE(S) ---
    (1) theta: parameter vector in (omega, alpha, beta) format
    (2) yt: time series for inference
    --- OUTPUT VARIABLE(S) ---
    (1) ScaledMinusLogLik: rescaled negative quasi log likelihood (without additive constants)
    """

    # Sample size
    T = len(yt)

    # Select parameters
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]

    #--- Estimate GARCH(1,1) process ---#
    # Initizalize recursions from unconditional variance
    sigma2t0 = max(omega/(1-alpha-beta), omega/(1-0.99))
    yt0 = np.sqrt(sigma2t0)

    # Recursion to reconstruct volatility process
    sigma2t = np.empty( (T, 1) ); sigma2t[:]= np.NaN
    for t in range(0, T):
        if t==0:
            sigma2t[t] = omega + alpha*yt0**2 + beta * sigma2t0
        else:
            sigma2t[t] = omega + alpha*yt[t-1]**2 + beta*sigma2t[t-1]
    ScaledMinusLogLik = (1/T)*np.sum( (yt**2)/sigma2t + np.log(sigma2t))

    # Return output
    return ScaledMinusLogLik

################################### FUNCTION: EstimateGARCH11 ###################################
def EstimateGARCH11(data, thetastart = np.array([0.05, 0.4, 0.4])):
    """
    DESCRIPTION: Estimate GARCH(1,1) process
    --- INPUT VARIABLE(S) ---
    (1) data: (Tx1) time series for GARCH estimation
    (2) thetastart (OPTIONAL): starting trial for nonlinear optimizer
    --- OUTPUT VARIABLE(S) ---
    (1) thetahat: estimate GARCH(1,1) parameters (omegahat, alphahat, betahat)
    (2) etahat: estimated innovations
    (3) AsymptCov: consistent estimator of the asymptotic covariance matrix
    """

    # ESTIMATION
    def ObjFunction(theta):
        funcval = GARCH11ObjFunc2Minimise(theta, data)
        return funcval
    MyBounds = [(0.01, np.inf), (0.01, np.inf), (0.01, np.inf)]
    OptimResult = minimize(ObjFunction, thetastart, method="L-BFGS-B", bounds=MyBounds)

    # Select OptimResult outputs
    thetahat = OptimResult["x"]
    invHessian = OptimResult['hess_inv'].todense()

    # Estimated innovations
    etahat = InnovationFilter(data, thetahat)
    kurtosis_etahat = kurtosis(etahat, fisher=False)

    # Asymptotic covariance matrix
    AsymptCov = (kurtosis_etahat-1)*invHessian

    return thetahat, etahat, AsymptCov

################################### FUNCTION: InnovationFilter ###################################
def InnovationFilter(yt, thetahat):
    """
    DESCRIPTION: Estimate innovations
    --- INPUT VARIABLE(S) ---
    (1) yt: (Tx1) time series for GARCH estimation
    (2) thetahat: MLE of GARCH(1,1)
    --- OUTPUT VARIABLE(S) ---
    (1) etahat: (Tx1) time series of estimated innovations
    """

    # Sample size
    T = len(yt)

    # Read quasi-MLEs from input
    omegahat = thetahat[0]
    alphahat = thetahat[1]
    betahat = thetahat[2]

    # Recursion to reconstruct volatility process
    sigma20hat = max(omegahat/(1-alphahat-betahat), omegahat/(1-0.99)); # Use max to prevent negative sigma20hat
    yt0hat = np.sqrt(sigma20hat);
    sigma2hat = np.empty( (T, 1) ); sigma2hat[:]= np.NaN
    for t in range(0, T):
        if t==0:
            sigma2hat[t] = omegahat + alphahat*yt0hat**2 + betahat * sigma20hat
        else:
            sigma2hat[t] = omegahat + alphahat*yt[t-1]**2 + betahat*sigma2hat[t-1]

    # Estimated innovations
    etahat = yt/np.sqrt(sigma2hat)

    return etahat
