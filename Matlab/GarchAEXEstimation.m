clc; clearvars; close all;
addpath("./functions/")

%--- Read Data ---%
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [2, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");
AEX = readtable("../AEX.csv", opts);
clear opts

%--- Pre-processing ---%
time = AEX.Date;
time(1) = []; % remove first observation because of differencing
yt = diff(log(AEX.Close));
fprintf("SUMMARY\n")
T = length(yt); fprintf("Time series length: %d\n", T);
NumberOfMissing = sum(isnan(yt)); fprintf("Number of missing observations: %d\n", NumberOfMissing);
fprintf("Percentage of missing observations: %5.2f\n\n", NumberOfMissing/T*100)

%--- Linearly Interpolate Missing Values ---%
ytLinearInterpollation = yt;
LastKnown = find(diff(isnan(yt))==1); % Find points before NaN
NextKnown = find(diff(isnan(yt))==-1)+1; % Find points after NaN
NumberOfMissingSequences = length(LastKnown);
for MissingSequence = 1:NumberOfMissingSequences
    OldValue = yt(LastKnown(MissingSequence));
    NewValue = yt(NextKnown(MissingSequence));
    MissingInterval = NextKnown(MissingSequence)-LastKnown(MissingSequence);
    for missingiter = 1:(MissingInterval-1)
        ytLinearInterpollation(LastKnown(MissingSequence)+missingiter) = OldValue + (missingiter/MissingInterval)*(NewValue-OldValue);
    end
end

%--- Plot Data ---%
figure(1)
hold on
plot([2114 2114], [-0.13 0.13],'--','LineWidth', 3, 'Color', 'r')
plot([5043 5043], [-0.13 0.13],'--','LineWidth', 3, 'Color', 'r')
plot(ytLinearInterpollation, 'Color',lines(1), 'LineWidth', 1);
hold off
set(gca,'fontsize', 15);
ylabel('AEX Log Return','FontSize',20)
axis([0 5400 -0.13 0.13])
xtickangle(70)
xticks([1 2114 5043 5400])
yticks([-0.1 -0.05 0 0.05 0.1])
xticklabels({'2000/08/08','2008/09/29','2020/03/12','2021/08/04'})
box on
set(gca, 'linewidth', 3)

%--- Estimate GARCH(1,1) Model ---%
fprintf("PARAMETER ESTIMATES\n")
Thetahat = EstimateGARCH11(ytLinearInterpollation, [0.015; 0.2; 0.74], 'none');
omegahat = Thetahat(1); alphahat = Thetahat(2); betahat = Thetahat(3);
fprintf("omegahat:\t%5.3g\n", omegahat)
fprintf("alphahat:\t%5.3f\n", alphahat)
fprintf("betahat:\t%5.3f\n\n", betahat)

%--- Reconstruct Volatility Process ---%
sigma20hat = omegahat/(1-alphahat-betahat);
yt0hat = sqrt(sigma20hat);
% Recursion to reconstruct volatility process
sigma2hat = NaN(T, 1);
for t = 1:T
    if t==1
        sigma2hat(t) = omegahat + alphahat*yt0hat^2 + betahat*sigma20hat;
    else
        sigma2hat(t) = omegahat + alphahat*ytLinearInterpollation(t-1)^2 + betahat*sigma2hat(t-1);
    end
end
figure(2)
hold on
plot([2114 2114], [-0.13 0.13],'--','LineWidth',3, 'Color', 'r')
plot([5043 5043], [-0.13 0.13],'--','LineWidth',3, 'Color', 'r')
plot(sigma2hat, 'Color', lines(1), 'LineWidth', 3);
hold off
set(gca,'fontsize', 15);
ylabel('$\hat\sigma_t^2$', 'FontSize', 25, 'Interpreter', 'latex')
axis([0 5400 -0.25E-3 4E-3])
xtickangle(70)
xticks([1 2114 5043 5400])
yticks([0 1E-3 2E-3 3E-3 4E-3])
xticklabels({'2000/08/08','2008/09/29','2020/03/12','2021/08/04'})
box on
set(gca, 'linewidth', 3)

%--- Autocorrelation Plots ---%
MaxLag = 60;
% Plot for levels
ACFLevels = autocorr(yt, 'NumLags', MaxLag);
figure(3)
stem((0:(length(ACFLevels)-1)), ACFLevels, 'o', 'filled', 'LineWidth', 1.5)
set(gca,'fontsize', 15);
xlabel('lag', 'FontSize', 25)
ylabel('sample autocorrelation', 'FontSize', 25)
axis([0 MaxLag -0.25 1])
hold on
plot([0 MaxLag], [1.96/sqrt(length(yt)) 1.96/sqrt(length(yt))], '--r', 'LineWidth', 2)
plot([0 MaxLag], [-1.96/sqrt(length(yt)) -1.96/sqrt(length(yt))], '--r', 'LineWidth', 2)
hold off
box on
set(gca, 'linewidth', 3)

% Plot for squares
ACFLevels = autocorr(yt.^2, 'NumLags', MaxLag);
figure(4)
stem((0:(length(ACFLevels)-1)), ACFLevels, 'o', 'filled', 'LineWidth', 1.5)
set(gca,'fontsize', 15);
xlabel('lag', 'FontSize', 25)
ylabel('sample autocorrelation', 'FontSize', 25)
axis([0 MaxLag -0.25 1])
hold on
plot([0 MaxLag], [1.96/sqrt(length(yt)) 1.96/sqrt(length(yt))], '--r', 'LineWidth', 2)
plot([0 MaxLag], [-1.96/sqrt(length(yt)) -1.96/sqrt(length(yt))], '--r', 'LineWidth', 2)
hold off
box on
set(gca, 'linewidth', 3)

%--- Further summary statistics ---%
fprintf("SUMMARY KURTOSIS RESULTS\n")
etahat = ytLinearInterpollation./sqrt(sigma2hat);
KurtResid = kurtosis(etahat);
fprintf("Sample kurtosis of yt:\t%5.4g\n", kurtosis(yt))
fprintf("Sample kurtosis of residuals: \t%5.4g\n", kurtosis(etahat))




