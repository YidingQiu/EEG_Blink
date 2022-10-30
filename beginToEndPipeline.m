% put RawData folder and raw_labled.mat in root dir together with this file
% run this file to get result of model
%% variables
rng(1);

slice = 512;                  % input slice length
wt = true;                    % whether the model use wavelet transform
maxEpochs = 30;               % training epoch
numTesting = 10;              % number of tests for optimised models 
BayesOptimiseFor = ['WTCNN',  % choose to run hyperparameter search for which models
                    '']; 
%% build path
% 'Img' is the place that store imgs for pixelclsificatin
% orthers are store as .mat
tDir = fullfile('DataSet','HyperparameterSearch','Train','Img');mkdir(tDir);
tDir = fullfile('DataSet','HyperparameterSearch','Test','Img');mkdir(tDir);
for i = 1:numTesting
    tDir = fullfile('DataSet','Test','Train',['Img',num2str(i)]);mkdir(tDir);    
end
tDir = fullfile('DataSet','Test','Test','Img');mkdir(tDir);
tDir = fullfile('DataSet','PreservedTrain','Img');mkdir(tDir);
optVarsPath = 'HyperParameterSearch\OptVars';  % optimised parameter here
if exist(optVarsPath) ~= 7
    mkdir(optVarsPath);
end
testResultPath = 'Test\Results';
if exist(testResultPath) ~= 7
    mkdir(testResultPath);
end
%% data preparation
Ds = dataPreparation();
if wt
    [dsTrain,dsTest] = imgDsPreparation();
else
    [trainDs,testDs] = dsPreparation();
end
disp('data set preparation done');
%% hyper parameter search 
%% WTCNN
if contains(BayesOptimiseFor,'WTCNN')
    optimVars = [
        %numBlocks,filterSizeC,filterSizeS, filterNum,
        %dropoutLayerRegularization,activation,learningrate
        optimizableVariable('batchSize',[8 32],'Type','integer')%[8 128]
        optimizableVariable('numBlocks',[1 5],'Type','integer')%[1 5]
        optimizableVariable('filterSizeC',[3 33],'Type','integer')%[3 33]
        optimizableVariable('filterSizeS',[3 33],'Type','integer')%[3 33]
        optimizableVariable('filterNum',[4 32],'Type','integer')%[4 32]   
        optimizableVariable('dropoutLayerRegularization',[0.00 0.2],'Type','real')
        optimizableVariable('activation',{'relu','tanh','swish'},'Type','categorical')    
        optimizableVariable('learningrate',[1e-4 1e-1],'Type','real',"Transform","log")];
    
    objectFunction = objectFunctionCNN(dsTrain, dsTest,  maxEpochs);
        
    % optimizer
    BayesObject = bayesopt( objectFunction,             optimVars, ...
                            'MaxTime',                  24*60*60, ...
                            'Verbose',                  1,...
                            "UseParallel",              false,...
                            'IsObjectiveDeterministic', true, ...
                            'ParallelMethod',           "clipped-model-prediction",...
                            'MaxObjectiveEvaluations',  30,...
                            'ExplorationRatio',         0.6,...
                            'AcquisitionFunctionName',  'expected-improvement-per-second-plus', ...
                            'OutputFcn',                {},...
                            'SaveFileName',             'wtcnnBayesoptResults.mat');
    
    optVars = bestPoint(BayesObject);
    
    save([optVarsPath '\WTCNNoptVars.mat'],"optVars","-mat");
end
%% test models
%% WTCNN
modelName = 'WTCNN';
% parameters
imSize=size(imread('DataSet\HyperparameterSearch\Train\Img\WTImg\0001_wt.jpg'));
numBlocks = optVars.numBlocks;
filterSizeC= floor((optVars.filterSizeC)/2)*2+1;
filterSizeS= floor((optVars.filterSizeS)/2)*2+1;
numFilter= optVars.filterNum;
dropoutLayerRegularization = optVars.dropoutLayerRegularization;
learningrate = optVars.learningrate;
activation=optVars.activation;
miniBatchSize=optVars.batchSize;
classes = ["blink","noBlink", "muscleArtifact"];
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate',learningrate, ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',0.75, ...
    'LearnRateSchedule','piecewise', ...
    'Plots','none',...                     % training-progress
    'GradientThreshold',1, ...
    'shuffle','every-epoch',...
    'Verbose',1,...
    'DispatchInBackground',true);
% tests
accuracy = {};
YPredResult = {};

for i = 1:numTesting
    % load data set
    YLable = load(fullfile('DataSet','Test','Test',['YTestTest' num2str(i) '.mat'])).YTestTest;
    TY = YLable{1};
    for k = 2:numel(YLable)
    TY = [TY;YLable{k}];
    end

    imdsTest = imageDatastore(fullfile('DataSet','Test','Test',['Img', num2str(i)],'WTImg'),FileExtensions=".jpg");
    testImgSize = [size(imread(imdsTest.Files{1,1})) 1 1];
    imgsTest = ones(testImgSize);
    for k =1:numel(imdsTest.Files)
        imgsTest = cat(4,imgsTest,reshape(imread(imdsTest.Files{k,1}), testImgSize));
    end
    imgsTest=imgsTest(:,:,:,2:end);

    YLable = load(fullfile('DataSet','HyperparameterSearch','Test','YSearchTest.mat')).YSearchTest;
    dataClassNames = ["blink","n/a", "muscle-artifact"];pixelLabelIds = 1:numel(dataClassNames);
    classNames=["blink","noBlink", "muscleArtifact"];
    imdsTrain = imageDatastore(['DataSet\Test\Train\Img' num2str(i) '\WTImg'],FileExtensions=".jpg");
    pxdsTrain = pixelLabelDatastore(['DataSet\Test\Train\Img' num2str(i) '\PLImg'],classNames,pixelLabelIds);
    
    imdsTest = imageDatastore(['DataSet\Test\Test\Img' num2str(i) '\WTImg'],FileExtensions=".jpg");
    pxdsTest = pixelLabelDatastore(['DataSet\Test\Test\Img' num2str(i) '\PLImg'],classNames,pixelLabelIds);
    
    dsTrain = combine(imdsTrain,pxdsTrain);
    layers = simpleCNNpixel(imSize,numBlocks,filterSizeC,filterSizeS, ...
    numFilter, dropoutLayerRegularization,classes,activation); 
    % train

    WTCNNnet = trainNetwork(dsTrain,layers,options);
    % evaluate
    [YPred,scores,allScores]=semanticseg(imgsTest,WTCNNnet);
    valError = 1-mean(YPred == TY','all');
    accuracy{end+1} = valError;
    YPredResult{end+1} = YPred;
%     disp(i);
    
end

result = table(accuracy, YPredResult);
save(['Test\Results\' modelName 'result.mat'],'result','-mat');


