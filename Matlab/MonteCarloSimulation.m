clear variables; clc; close all;
addpath("./functions")

% Simulation parameters for GARCH(1,1) Monte Carlo
theta = [0.1; 0.05; 0.8];
T = 1E4;
Nsim = 1E3;
rng(1)

% Initialise matrices
ParaEst = NaN(3, Nsim);
tstat = NaN(3, Nsim);

tic
% Estimate GARCH(1,1) data series
for simiter = 1:Nsim

    % Report progress
    if (mod(simiter,1E2)) == 0
        fprintf('Iteration %5d out of %5d \n', simiter, Nsim);
    end

    % Generate GARCH(1,1) data set
    data = GenerateGARCH11(theta, T);

    % Estimation
    [est, ~, AsymptCovMatrix] = EstimateGARCH11(data, theta);
    ParaEst(:, simiter) = est;
    tstat(:, simiter) = sqrt(T)*(est-theta)./sqrt( diag(AsymptCovMatrix) );

    % Set complex outcomes to zero (Nb. a flat objective function can cause a negative asymptotic variance)
    for iter = 1:size(ParaEst, 1)
        if ~isreal(tstat(iter, simiter))
            disp("Negative variance encountered")
            tstat(iter, simiter) = NaN;
        end
    end
end
toc

figure(1)
histogram(tstat(2, :), 20, 'Normalization', 'pdf') % Ignores NaN
hold on
xlist = -5:0.05:5;
ylist = normpdf(xlist);
plot(xlist, ylist, 'LineWidth', 2)
hold off
axis([-5 5 0 0.5])

