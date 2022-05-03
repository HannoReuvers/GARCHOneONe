clear variables; clc; close all;
addpath("./functions")


% Parameters of GARCH(1,1) specification
theta = [0.03; 0.05; 0.9];
T = 5000;

% Generate GARCH(1,1)
yt = GenerateGARCH11(theta, T);

% Estimate GARCH(1,1)
thetahat = EstimateGARCH11(yt);

plot(yt)

GARCH11ObjFunc2Minimise(theta, yt)
mean(yt.^2)

