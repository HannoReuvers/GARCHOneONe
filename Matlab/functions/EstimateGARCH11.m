function [thetahat, etahat, AsymptCov] = EstimateGARCH11(data, thetastart, OptimizerDisplayOption)
%% DESCRIPTION: Estimate GARCH(1,1) process
%---INPUT VARIABLE(S)---
%   (1) data: (Tx1) time series for GARCH estimation
%   (2) thetastart (OPTIONAL): starting trial for nonlinear optimizer
%   (3) OptimizerDisplayOption: additional options to pass to fmincon
%---OUTPUT VARIABLE(S)---
%   (1) thetahat: estimate GARCH(1,1) parameters (omegahat, alphahat, betahat)
%   (2) etahat: estimated innovations
%   (3) AsymptCov: consistent estimator of the asymptotic covariance matrix
%   of the MLE
    
    % Default starting parameters
    if nargin < 2
        thetastart = [0.05; 0.4; 0.4];
        OptimizerDisplayOption = 'none';
    elseif nargin < 3
        OptimizerDisplayOption = 'none';
    end
    
    %--- ESTIMATION ---%
    options = optimoptions(@fmincon, 'Display', OptimizerDisplayOption, 'Algorithm', 'interior-point');
    ObjFunction = @(x) GARCH11ObjFunc2Minimise(x, data);
    % Parameters
    [thetahat, ~, ~, ~, ~, ~, hessian] = fmincon(ObjFunction, thetastart, [], [], [], [], [1E-7; 0.01; 0.01], [], [], options);
    % Estimated innovations
    etahat = InnovationFilter(data, thetahat);
    kurtosis_etahat = kurtosis(etahat);
    % Asymptotic covariance matrix
    AsymptCov = (kurtosis_etahat-1)*inv(hessian);
end

function [etahat] = InnovationFilter(yt, thetahat)
%% DESCRIPTION: Estimate innovations
%---INPUT VARIABLE(S)---
%   (1) yt: (Tx1) time series for GARCH estimation
%   (2) thetahat: MLE of GARCH(1,1)
%---OUTPUT VARIABLE(S)---
%   (1) etahat: (Tx1) time series of estimated innovations

    % Sample size
    T = length(yt);

    % Read quasi-MLEs from input
    omegahat = thetahat(1);
    alphahat = thetahat(2);
    betahat = thetahat(3);

    % Recursion to reconstruct volatility process
    sigma20hat = max(omegahat/(1-alphahat-betahat), omegahat/(1-0.99)); % Use max to prevent negative sigma20hat
    yt0hat = sqrt(sigma20hat);
    sigma2hat = NaN(T, 1);
    for t = 1:T
        if t==1
            sigma2hat(t) = omegahat + alphahat*yt0hat^2 + betahat*sigma20hat;
        else
            sigma2hat(t) = omegahat + alphahat*yt(t-1)^2 + betahat*sigma2hat(t-1);
        end
    end

    % Estimated innovations
    etahat = yt./sqrt(sigma2hat);
end

