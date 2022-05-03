function [ScaledMinusLogLik] = GARCH11ObjFunc2Minimise(theta, yt)
%% DESCRIPTION: Objective function to MINIMISE for Gaussian QMLE
%---INPUT VARIABLE(S)---
%   (1) theta: parameter vector in (omega, alpha, beta) format
%   (2) yt: time series for inference
%---OUTPUT VARIABLE(S)---
%   (1) ScaledMinusLogLik: rescaled negative quasi log likelihood (without additive constants)

    % Sample size
    T = length(yt);

    % Select parameters
    omega = theta(1);
    alpha = theta(2);
    beta = theta(3);

    %--- Estimate GARCH(1,1) process ---%
    % Initialize recursions from unconditional variance
    sigma2t0 = omega/(1-alpha-beta);
    yt0 = sqrt(sigma2t0);

    % Recursion to reconstruct volatility process
    sigma2t = NaN(T, 1);
    for t = 1:T
        if t==1
            sigma2t(t) = omega + alpha*yt0^2 + beta*sigma2t0;
        else
            sigma2t(t) = omega + alpha*yt(t-1)^2 + beta*sigma2t(t-1);
        end
    end
    ScaledMinusLogLik = mean( (yt.^2)./sigma2t +log(sigma2t) );
end

