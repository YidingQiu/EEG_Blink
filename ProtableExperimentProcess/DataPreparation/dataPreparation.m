function Ds = dataPreparation(slice,fs,dsFactor,numTesting,ratioValidationTest,ratioPreserveTrain,frontElectrodesLabels)
    arguments
        slice = 512,
        fs = 512,
        dsFactor = 4,
        numTesting = 10,
        ratioValidationTest = 0.2,
        ratioPreserveTrain = 0.1
        frontElectrodesLabels = ["Fp1", "Fp2", "Fz", "AF3", "AF4", "F7", "F8"]; % 
    end

    %% signal and label read in, preprocess
    % read lables
    rng(1);
    ds = load('raw_labled.mat', 'ls');
    ds = ds.ls.Labels;
    lables = ds.detect_blinks;    
    % read filtered EEG data
    currentDir = dir;
    currentDir(1:2) = [];
    for i = 1:numel(currentDir)
        if strcmp(currentDir(i).name,'RawData')
            %disp('get RawData')
            pathToEEG = fullfile(currentDir(i).folder,currentDir(i).name);
            break
        end
    end
    eegFiles = dir(pathToEEG);
    eegFiles(1:2) = [];
    Ds = {};
    
    
    categroies = {'blink', 'n/a', 'muscle-artifact'};
    for k = 1:31
        tokens = strsplit(ds.Row{k}, '\');
        fileName = tokens{end};
        eegTable = readtable([pathToEEG,'\',eegFiles(k).name]);
        filteredSignals = passbandBlinks(eegTable(:, contains(eegTable.Properties.VariableNames, frontElectrodesLabels)), fs);
        filteredSignals = array2table(resample(filteredSignals.Variables, 1, dsFactor),  'VariableNames', filteredSignals.Properties.VariableNames);
        roi = ds(k,:).detect_blinks{1};
        roi(:, 1).Variables = ceil(roi(:, 1).Variables/dsFactor);
        masked = getmask({filteredSignals, roi});
        
        Ds{k} = masked; 
        Ds{k}{2} = categorical(Ds{k}{2},categroies);
    end
    disp("filtering done");

    %% data splition and write file

    ratioSearchTrain = (1-ratioValidationTest-ratioPreserveTrain)*0.8;
    ratioSearchTest = (1-ratioValidationTest-ratioPreserveTrain)*(1-0.8);
    ratioTestSave = ratioPreserveTrain+ratioSearchTest;
    ratios = [ratioSearchTrain,ratioTestSave,ratioValidationTest];
    
    XSearchTrain = {};
    YSearchTrain = {};
    XTestSave = {};
    YTestSave = {};
    XValidationTest = {};
    YValidationTest = {};
    
    
    for k = 1:length(Ds)
        [indexSearchTrain,indexTestSave,indexValidationTest] = splitSignalIntoSeveralSegments(Ds{k}{1}, ratios, slice/8, slice);
    
        for l = 1:size(indexSearchTrain, 1)
            XSearchTrain{end+1} = customNormalize(Ds{k}{1}(indexSearchTrain(l,:), :)).Variables';
            YSearchTrain{end+1} = Ds{k}{2}(indexSearchTrain(l,:))';
        end
        for l = 1:size(indexTestSave, 1)
            XTestSave{end+1} = customNormalize(Ds{k}{1}(indexTestSave(l,:), :)).Variables';
            YTestSave{end+1} = Ds{k}{2}(indexTestSave(l,:))';
        end
        for l = 1:size(indexValidationTest, 1)
            XValidationTest{end+1} = customNormalize(Ds{k}{1}(indexValidationTest(l,:), :)).Variables';
            YValidationTest{end+1} = Ds{k}{2}(indexValidationTest(l,:))';
        end 
    end
    disp("Splition done");
    
    mkdir(fullfile('DataSet','HyperparameterSearch','Train'));
    save(fullfile('DataSet','HyperparameterSearch','Train','XSearchTrain.mat'),"XSearchTrain","-mat");
    save(fullfile('DataSet','HyperparameterSearch','Train','YSearchTrain.mat'),"YSearchTrain","-mat");

    mkdir(fullfile('DataSet','Test','Test'));
    save(fullfile('DataSet','Test','Test','XValidationTest.mat'),"XValidationTest","-mat");
    save(fullfile('DataSet','Test','Test','YValidationTest.mat'),"YValidationTest","-mat");

    
    [searchTestIndex,preserveTrainIndex,~]=dividerand(numel(XTestSave),ratioSearchTest/ratioTestSave,ratioPreserveTrain/ratioTestSave,0);
    
    XSearchTest = XTestSave(searchTestIndex);
    YSearchTest = YTestSave(searchTestIndex);
    XPreserveTrain = XTestSave(preserveTrainIndex);
    YPreserveTrain = YTestSave(preserveTrainIndex);
    
    mkdir(fullfile('DataSet','HyperparameterSearch','Test'));
    save(fullfile('DataSet','HyperparameterSearch','Test','XSearchTest.mat'),"XSearchTest","-mat");
    save(fullfile('DataSet','HyperparameterSearch','Test','YSearchTest.mat'),"YSearchTest","-mat");

    mkdir(fullfile('DataSet','PreservedTrain'));
    save(fullfile('DataSet','PreservedTrain','XPreserveTrain.mat'),"XPreserveTrain","-mat");
    save(fullfile('DataSet','PreservedTrain','YPreserveTrain.mat'),"YPreserveTrain","-mat");
    disp("HyperparameterSearch dataset done");
    
    
    % combine the training set and an extra training set (70%+10%) and then 
    % randomly sample out of it (70%) 10 times 
    XTestTrainBase = [XSearchTrain,XTestSave];
    YTestTrainBase = [YSearchTrain,YTestSave];
    mkdir(fullfile('DataSet','Test','Test'));
    mkdir(fullfile('DataSet','Test','Train'));

    for i=1:numTesting
        rng(i);
        [testTrainIndex,addTestTestIndex,~]=dividerand(numel(XTestTrainBase),0.8,0.2,0);
        
        XTestTrain = XTestTrainBase(testTrainIndex);
        YTestTrain = YTestTrainBase(testTrainIndex);
        XTestTest = [XTestTrainBase(addTestTestIndex),XValidationTest];
        YTestTest = [YTestTrainBase(addTestTestIndex),YValidationTest];
        
        save(fullfile('DataSet','Test','Test',['XTestTest',num2str(i),'.mat']),"XTestTest","-mat");
        save(fullfile('DataSet','Test','Test',['YTestTest',num2str(i),'.mat']),"YTestTest","-mat"); 
        
        save(fullfile('DataSet','Test','Train',['XTestTrain',num2str(i),'.mat']),"XTestTrain","-mat");
        save(fullfile('DataSet','Test','Train',['YTestTrain',num2str(i),'.mat']),"YTestTrain","-mat");

    end
    disp("Test dataset done");
end