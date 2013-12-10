%This Demo shows how to run R2LMTL
clear;clc;
%%Read the data
path = [pwd,'/Data/ionophsere'];
addpath([pwd,'/Functions']);

%%We will set all the hyperparameters here
%Number of metric
parameters.NumMa_K = 2;
%Regularization value lambda
parameters.lambda = 100;
%Step length of PSD
parameters.t0 = 1e-6;
%Number of steps of PSD for each metric
parameters.iter = 800;
%Number of epoches of two steps
parameters.epoch = 10;
%Number of k-nearest neighbors when testing
parameters.kneigh = 5;



%%Run the algorithm
accu = R2LMTL(path,parameters);
