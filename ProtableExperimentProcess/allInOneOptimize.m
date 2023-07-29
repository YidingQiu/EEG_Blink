% this is Bayes Optimization process for all models

%% setting
rng(1);
maxEpochs = 3;%30;
numMaxObjectiveEvaluation = 10 ;%100;
numChannel = 5:-2:1;



if ~isfolder(fullfile('HyperParameterSearch','OptVars'))
    mkdir(fullfile('HyperParameterSearch','OptVars'));
end

%% load dataset

XTrain = load(fullfile('DataSet','HyperparameterSearch','Train','XSearchTrain.mat')).XSearchTrain;
YTrain = load(fullfile('DataSet','HyperparameterSearch','Train','YSearchTrain.mat')).YSearchTrain;

XTest = load(fullfile('DataSet','HyperparameterSearch','Test','XSearchTest.mat')).XSearchTest;
YTest = load(fullfile('DataSet','HyperparameterSearch','Test','YSearchTest.mat')).YSearchTest;

categroies = {'closing', 'opening', 'n/a', 'muscle-artifact'};
[YTrain,locationTrain] = generateOpeningClosing(XTrain, YTrain,categroies);
[YTest,locationTest] = generateOpeningClosing(XTest, YTest,categroies);
[trainDs,testDs] = dsPreparation();


%% bayes optimize

for i = numChannel

    XTrain = selectChannels(XTrain, (1:i));
    XTest = selectChannels(XTest, (1:i));
    trainDs = {XTrain, YTrain};
    testDs = {XTest, YTest};
    
    %% CovMix
    optimVars = [
        % Depth HiddenDimension paddingSizeS  dropoutLayerRegularization 
        optimizableVariable('Depth',[1 5],'Type','integer')
        optimizableVariable('numFilters',[1 8],'Type','integer')
        optimizableVariable('HiddenDimension',[2 32],'Type','integer')
        optimizableVariable('filterSizeS',[1 5],'Type','integer')
        optimizableVariable('dropoutLayerRegularization',[0.00 0.1],'Type','real')
        optimizableVariable('miniBatchSize',[8 128],'Type','integer')    
        optimizableVariable('learningrate',[1e-4 1e-1],'Type','real',"Transform","log")];

    objectFunction = objectFunctionCovMix(trainDs, testDs,  maxEpochs);
    BayesObject = createBayesOpt(objectFunction, optimVars,numMaxObjectiveEvaluation);
    [bestError, trainedNet] = getResult(BayesObject,'CovMix',i);
    %% TCN
    optimVars = [
        %numFilters, filterSize, numBlocks, dropoutFactor
        optimizableVariable('numFilters',[2 32],'Type','integer')
        optimizableVariable('filterSize',[3 33],'Type','integer')
        optimizableVariable('numBlocks',[1 5],'Type','integer')
        optimizableVariable('dropoutFactor',[0.00 0.1],'Type','real')
        optimizableVariable('miniBatchSize',[8 128],'Type','integer')    
        optimizableVariable('learningrate',[1e-4 1e-1],'Type','real',"Transform","log")];
        
    objectFunction = objectFunctionTCN(trainDs, testDs,  maxEpochs);
    BayesObject = createBayesOpt(objectFunction, optimVars,numMaxObjectiveEvaluation);
    [bestError, trainedNet] = getResult(BayesObject,'TCN',i);
    %% LSTM & biLSTM
    optimVars = [
        optimizableVariable('miniBatchSize',[8 128],'Type','integer')%[8 128]
        optimizableVariable('learningrate',[1e-4 1e-1],'Type','real',"Transform","log")
        optimizableVariable('numBlocks',[1 5],'Type','integer')%[1 5]
        optimizableVariable('numHiddenUnits',[1 128],'Type','integer')];
        
    objectFunction = objectFunctionLSTM(trainDs, testDs,  maxEpochs);
    BayesObject = createBayesOpt(objectFunction, optimVars,numMaxObjectiveEvaluation);
    [bestError, trainedNet] = getResult(BayesObject,'LSTM',i);
    objectFunction = objectFunctionbiLSTM(trainDs, testDs,  maxEpochs);
    BayesObject = createBayesOpt(objectFunction, optimVars,numMaxObjectiveEvaluation);
    [bestError, trainedNet] = getResult(BayesObject,'biLSTM',i);
    %% WT-CNN
    optimVars = [
        %numBlocks,filterSizeC,filterSizeS, filterNum,
        %dropoutLayerRegularization,activation,learningrate
        optimizableVariable('batchSize',[8 128],'Type','integer')%[8 128]
        optimizableVariable('numBlocks',[1 5],'Type','integer')%[1 5]
        optimizableVariable('filterSizeC',[3 33],'Type','integer')%[3 33]
        optimizableVariable('filterSizeS',[3 33],'Type','integer')%[3 33]
        optimizableVariable('filterNum',[4 32],'Type','integer')%[4 32]   
        optimizableVariable('dropoutLayerRegularization',[0.00 0.1],'Type','real')
        optimizableVariable('learningrate',[1e-4 1e-1],'Type','real',"Transform","log")];

    objectFunction = objectFunctionCNN(trainDs, testDs,  maxEpochs,i);
    BayesObject = createBayesOpt(objectFunction, optimVars,numMaxObjectiveEvaluation);
    [bestError, trainedNet] = getResult(BayesObject,'WTCNN',i);
end
disp("optimization done. optimizedVariables in HyperParameterSearch/OptVars")
%%
function BayesObject = createBayesOpt(objectFunction, optimVars,numMaxObjectiveEvaluation)
    BayesObject = bayesopt( objectFunction, optimVars, ...
                            'MaxTime',                  6*60*60, ...
                            'Verbose',                  1,...
                            'UseParallel',              false,...
                            'IsObjectiveDeterministic', true, ...
                            'ParallelMethod',           'clipped-model-prediction',...
                            'MaxObjectiveEvaluations',  numMaxObjectiveEvaluation,...
                            'ExplorationRatio',         0.6,...
                            'AcquisitionFunctionName',  'expected-improvement-per-second-plus');
end
%% 
function [valError,net] = getResult(BayesObject,modelName,numChannel)
    optVars = bestPoint(BayesObject);

    save(['HyperParameterSearch\OptVars\', modelName, num2str(numChannel), 'optVars.mat'],"optVars","-mat");
    [valError,~,net, YPred, YTest] = BayesObject.ObjectiveFcn(optVars,1);
end
