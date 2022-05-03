function [yt] = GenerateGARCH11(theta, T)
%% DESCRIPTION: Simulate GARCH(1,1) process
%---INPUT VARIABLE(S)---
%   (1) theta: parameter vector in (omega, alpha, beta) format
%   (2) T: sample size
%---OUTPUT VARIABLE(S)---
%   (1) yt: simulated GARCH(1,1) time series

    % Select parameters
    omega = theta(1);
    alpha = theta(2);
    beta = theta(3);

    %--- Simulate GARCH(1,1) process ---%
    % Initialize recursions from unconditional variance
    sigma2t0 = omega/(1-alpha-beta);
    yt0 = sqrt(sigma2t0);

    % Recursion to simulate GARCH(1,1) with Gaussian innovations
    sigma2t = NaN(T, 1);
    yt = NaN(T, 1);
    for t = 1:T
        % Update volatility process
        if t==1
            sigma2t(t) = omega + alpha*yt0^2 + beta*sigma2t0;
        else
            sigma2t(t) = omega + alpha*yt(t-1)^2 + beta*sigma2t(t-1);
        end

        % Update data
        yt(t) = sqrt(sigma2t(t))*normrnd(0, 1);
    end
end

