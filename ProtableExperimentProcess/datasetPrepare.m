% put RawData folder and raw_labled.mat in root dir together with this file
% 
[currentScriptPath,~,~] = fileparts(mfilename('fullpath'));
subpath = genpath(currentScriptPath);
addpath(subpath); % path set up
%% variables
rng(1);

slice = 512;                  % input slice length               
maxEpochs = 30;               % training epoch
numTesting = 10;              % number of tests for optimised models 

%% build path
optVarsPath = fullfile('HyperParameterSearch','OptVars');  % optimised parameter here
if exist(optVarsPath) ~= 7
    mkdir(optVarsPath);
end
testResultPath = fullfile('Test','Results');
if exist(testResultPath) ~= 7
    mkdir(testResultPath);
end
%% data preparation
Ds = dataPreparation();

[trainDs,testDs] = dsPreparation();

disp('data set preparation done');
