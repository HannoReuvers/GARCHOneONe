import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.optimize import minimize
from scipy.stats import kurtosis
import timeit

# Load function library
CurrentDirectory = os.path.dirname(__file__)
print(CurrentDirectory)
LibraryScriptLocation = os.path.join(CurrentDirectory, 'functions', 'GARCH11Functions.py')
exec(open(LibraryScriptLocation).read())

# Simulation parameters for GARCH(1,1) Monte Carlo
theta = np.array([0.1, 0.05, 0.8])
T = 10000
Nsim = 1000
random.seed(1)

# Initialise matrices
ParaEst = np.empty( (3, Nsim) ); ParaEst[:]= np.NaN
tstat = np.empty( (3, Nsim) ); tstat[:]= np.NaN

StartTimer = timeit.default_timer()
# Estimate GARCH(1,1) data series
for simiter in range(0, Nsim):

    # Report progress
    if simiter%1E2==0:
        print("Iteration", simiter, "out of", Nsim)

    # Generate GARCH(1,) data set
    yt = GenerateGARCH11(theta, T)

    # Estimation
    est, etahat, AsymptCovMatrix = EstimateGARCH11(yt, theta)
    ParaEst[:, simiter] = est
    tstat[:, simiter] = np.sqrt(T)*(est-theta)/np.sqrt( np.diag(AsymptCovMatrix) )

StopTimer = timeit.default_timer()

# Report time
print("Elapsed time is", StopTimer-StartTimer, "seconds.")

# Plot histogram of result
plt.hist(x=tstat[1,:], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.show()
