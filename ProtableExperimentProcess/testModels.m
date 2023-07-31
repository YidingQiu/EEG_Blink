% get accuracy on test sets

%% load parameters

pathToOptVars = fullfile('HyperParameterSearch', 'OptVars');



%WTCNN
optVarsFile = fullfile(pathToOptVars, 'WTCNNoptVars.mat');
optVars = load(optVarsFile).optVars;
numBlocks = optVars.numBlocks;
filterSizeC= floor((optVars.filterSizeC)/2)*2+1;
filterSizeS= floor((optVars.filterSizeS)/2)*2+1;
numFilter= optVars.filterNum;
dropoutLayerRegularization = optVars.dropoutLayerRegularization;
learningrate = optVars.learningrate;
activation=optVars.activation;
miniBatchSize=optVars.batchSize;
classes = ["blink","noBlink", "muscleArtifact"];

%% training
rng(1);
numTesting = 10;
maxEpochs = 30;
numResponse = 4;

for i = 5:-2:1
    
    % LSTM/biLSTM
    optVarsFile = fullfile(pathToOptVars, 'biLSTMoptVars.mat');
    optVars = load(optVarsFile).optVars;
    miniBatchSize = optVars.miniBatchSize;
    learningrate = optVars.learningrate;
    numFeatures = i;
    numBlocks = optVars.numBlocks;
    numHiddenUnits = optVars.numHiddenUnits;    
    
    % LSTM
    modelName = 'LSTM';

    layers = constructLSTM(numFeatures,numBlocks,numHiddenUnits,numResponse,false); 
    
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateDropPeriod',10, ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateSchedule','piecewise', ...
        'Plots','none',...
        'GradientThreshold',1, ...
        'shuffle','every-epoch',...
        'Verbose',0,...
        'DispatchInBackground',true);
    
    [accuracy, confusionmats, pred] = evaluateAndSave(modelName, numTesting, layers, options);
    
    % biLSTM
    modelName = 'biLSTM';
    
    layers = constructbiLSTM(numFeatures,numBlocks,numHiddenUnits,numResponse,false);     
    
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateDropPeriod',10, ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateSchedule','piecewise', ...
        'Plots','none',...
        'GradientThreshold',1, ...
        'shuffle','every-epoch',...
        'Verbose',0,...
        'DispatchInBackground',true);
    
    [accuracy, confusionmats, pred] = evaluateAndSave(modelName, numTesting, layers, options);
        

    % TCN
    optVarsFile = fullfile(pathToOptVars, ['TCN' num2str(i) 'optVars.mat']);
    optVars = load(optVarsFile).optVars;
    numFilters = optVars.numFilters;
    filterSize = optVars.filterSize;
    numBlocks = optVars.numBlocks;
    dropoutFactor = optVars.dropoutFactor;
    miniBatchSize = optVars.miniBatchSize;
    learningrate = optVars.learningrate;

    % TCN
    modelName = 'TCN';
    layers = constructTCN(size(XTrain{1}, 1), numFilters, filterSize, numBlocks, dropoutFactor);
    
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateDropPeriod',10, ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateSchedule','piecewise', ...
        'Plots','none',...
        'GradientThreshold',1, ...
        'shuffle','every-epoch',...
        'Verbose',0,...
        'DispatchInBackground',true);
    
    [accuracy, confusionmats, pred] = evaluateAndSave(modelName, numTesting, layers, options);
    
    % WTCNN
    modelName = 'WTCNN';
    
    classes = ['closing', 'opening', 'n/a', 'muscle-artifact'];
    
    layers = constructCWTCNN(i,numBlocks,filterSizeC,filterSizeS, numFilter, dropoutLayerRegularization,classes,"relu");
        
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',batchSize, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateDropPeriod',5, ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateSchedule','piecewise', ...
        'Plots','none',...
        'GradientThreshold',1, ...
        'shuffle','every-epoch',...
        'Verbose',0,...
        'DispatchInBackground',true);
    
    [accuracy, confusionmats, pred] = evaluateAndSave(modelName, numTesting, layers, options);
    
    % ConvMixer


    
end



figure;
confusionchart(load(['Test\Results\' modelName 'result.mat']).result.confusionmats{1}, ...
    {'blink' 'n/a' 'muscle-artifact'},'Normalization','column-normalized','RowSummary','row-normalized');

Blink detect accuracy
i = 1;
% i=errorList(2);
Ypred = YPred{i};
Ytest = YTest{i};
Xtest = XTest{i};
Xdif = diff(Xtest')';

figure;

subplot(2,1,1);plot(Xtest');xlim([1 512])
title("Input signals")

% subplot(4,1,2);plot(Xdif');xlim([1 512])
% title("Differential signals")

subplot(2,1,2);plot(Ytest);xlim([1 512])
title("Ground truth labels")

% subplot(4,1,4);plot(Ypred);xlim([1 512])
% title("Predict labels")


% blinkLocate(Xtest,Ytest),blinkLocate(Xtest,Ypred)YTest
valError = 0;
for k = 1:length(YPred)
    valError = valError + mean(YPred{k} == YTest{k});
end
accuracy = valError/length(YPred)
TY = [];
Y = [];
YTest = YTest.';
for k = 1:numel(YTest)
    TY = [TY;YTest{k}];
    Y = [Y;YPred{k}];
end
confusionMat = confusionmat(TY(:),Y(:));
 confusionchart( confusionMat,...
    {'opening' 'closing' 'muscle-artifact' 'n/a'},'Normalization','column-normalized','RowSummary','row-normalized');



% OC test
% groundTruth = {};
% predicted = {};
% TP = 0;FP = 0;FN=0;
% errorList = [];
% for i = 1:numel(YTest)
%     gt = findBlinkFromOpeningClosing(YTest{i});
%     p = findBlinkFromOpeningClosing(YPred{i});
%     gt = gt{1};p=p{1};
%     groundTruth{end+1} = gt;
%     predicted{end+1} = p;
% 
%     TP = TP + numel(intersect(gt,p));
%     if (numel(gt) > numel(p))
%         FN = FN + numel(gt) - numel(p);
%     else
%         FP = FP + numel(p) - numel(gt);
%     end
%     if (numel(gt) ~= numel(p))
%         errorList(end+1) = i;
%     end        
% end 
% TP,FP,FN,errorList
% Precision=TP/(TP+FP),Recall=TP/(TP+FN)


% blink test

for i=1:10

    XTrain = load(['DataSet\Test\Train\XTestTrain' num2str(i) '.mat']).XTestTrain;
    YTrain = load(['DataSet\Test\Train\YTestTrain' num2str(i) '.mat']).YTestTrain;
    XTest = load(['DataSet\Test\Test\XTestTest' num2str(i) '.mat']).XTestTest;
    YTest = load(['DataSet\Test\Test\YTestTest' num2str(i) '.mat']).YTestTest; 

groundTruth = {};
predicted = {};
TP = 0;FP = 0;FN=0;
errorList = [];
YPred = Pred{1};
YPred(YPred=='noBlink')='n/a';

for k = 1:numel(YTest)
    gt = blinkLocate(XTest{k},YTest{k});
    p = blinkLocate(XTest{k},YPred(:,k));
    groundTruth{end+1} = gt;
    predicted{end+1} = p;
    
    TP = TP + sum(ismembertol(gt,p,50));%numel(intersect(gt,p));
    if (numel(gt) > numel(p))
        FN = FN + numel(gt) - numel(p);
    else
        FP = FP + numel(p) - numel(gt);
    end
    if (numel(gt) ~= numel(p))
        errorList(end+1) = k;
    end        
end 
disp(i);
TP,FP,FN,errorList
Precision=TP/(TP+FP),Recall=TP/(TP+FN)
end

for p = 0:5
tic;
i = 1;
[location,timePassed,mask] = detectBlinks(rawDs{1}.Variables',512,'1d-CNN',shiftFactor=p,plot=0,output={'location','time','mask'});
toc
end
%% dataset loader
function [XTrain, YTrain, XTest, YTest] = loadData(i)
    XTrainFile = fullfile('DataSet', 'Test', 'Train', ['XTestTrain' num2str(i) '.mat']);
    YTrainFile = fullfile('DataSet', 'Test', 'Train', ['YTestTrain' num2str(i) '.mat']);
    XTestFile = fullfile('DataSet', 'Test', 'Test', ['XTestTest' num2str(i) '.mat']);
    YTestFile = fullfile('DataSet', 'Test', 'Test', ['YTestTest' num2str(i) '.mat']);
    
    XTrain = load(XTrainFile).XTestTrain;
    YTrain = load(YTrainFile).YTestTrain;
    XTest = load(XTestFile).XTestTest;
    YTest = load(YTestFile).YTestTest;
end
%%
function [accuracy, confusionmats, pred] = evaluateAndSave(modelName, numTesting, layers, options)
    accuracy = {};
    confusionmats = {};
    pred = {};
    for i = 1:numTesting
        [XTrain, YTrain, XTest, YTest] = loadData(i);
        net = trainNetwork(XTrain,YTrain,layers,options);
        % evaluate
        YPred = classify(net, XTest, "MiniBatchSize",miniBatchSize);
        valError = 0;
        for k = 1:length(YPred)
            valError = valError + mean(YPred{k} == YTest{k});
        end
        accuracy{end+1} = valError/length(YPred);
        TY = [];
        Y = [];
        YTest = YTest.';
        for k = 1:numel(YTest)
            TY = [TY;YTest{k}];
            Y = [Y;YPred{k}];
        end
        confusionMat = confusionmat(TY(:),Y(:));
        pred{end+1} = YPred; 
        confusionmats{end+1} = confusionMat;
        if i == 1
            save(fullfile('Models', 'trainedModels', [modelName 'net.mat']), 'net', '-mat');
        end
    end
    result = table(accuracy, confusionmats);
    save(fullfile('Test', 'Results', [modelName 'result.mat']), 'result', '-mat');
    save(fullfile('Test', 'Results', [modelName 'pred.mat']), 'pred', '-mat');
end

