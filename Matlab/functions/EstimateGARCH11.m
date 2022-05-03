function [thetahat] = EstimateGARCH11(data, thetastart, OptimizerDisplayOption)
%% DESCRIPTION: Estimate GARCH(1,1) process
%---INPUT VARIABLE(S)---
%   (1) data: (Tx1) time series for GARCH estimation
%   (2) thetastart (OPTIONAL): starting trial for nonlinear optimizer
%---OUTPUT VARIABLE(S)---
%   (1) thetahat: estimate GARCH(1,1) parameters (omegahat, alphahat, betahat)
    
    % Default starting parameters
    if nargin < 2
        thetastart = [0.05; 0.4; 0.4];
        OptimizerDisplayOption = 'none';
    end
    
    % Objective function
    options = optimoptions(@fmincon, 'Display', OptimizerDisplayOption, 'Algorithm', 'interior-point');
    ObjFunction = @(x) GARCH11ObjFunc2Minimise(x, data);
    thetahat = fmincon(ObjFunction, thetastart, [], [], [], [], (1E-7)*ones(3,1), [], [], options);
end

