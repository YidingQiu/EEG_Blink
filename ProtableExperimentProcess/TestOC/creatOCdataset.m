function [XTrain,XTest,YTrain,YTest] = creatOCdataset(i)
    XTrain = load(fullfile('DataSet', 'Test', 'Train', ['XTestTrain' num2str(i) '.mat'])).XTestTrain;
    YTrain = load(fullfile('DataSet', 'Test', 'Train', ['YTestTrain' num2str(i) '.mat'])).YTestTrain;
    XTest = load(fullfile('DataSet', 'Test', 'Test', ['XTestTest' num2str(i) '.mat'])).XTestTest;
    YTest = load(fullfile('DataSet', 'Test', 'Test', ['YTestTest' num2str(i) '.mat'])).YTestTest;
    %translate 'blink' to 'opening' and 'closing'
    categroies = {'closing', 'opening', 'n/a', 'muscle-artifact'};
    [YTrain,~] = generateOpeningClosing(XTrain, YTrain,categroies);
    [YTest,~] = generateOpeningClosing(XTest,YTest,categroies);
    locationTrain = YTrain{2};
    locationTest = YTest{2};
    YTrain = YTrain{1};
    YTest = YTest{1};
end