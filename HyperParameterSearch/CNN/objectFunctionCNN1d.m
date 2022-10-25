function ObjFcn = objectFunctionTCN(Traindata, Testdata,  Maxepochs)
ObjFcn = @valErrorFun;
    function [valError,emp, net, YPred, YTest] = valErrorFun(optVars,mviz)
        emp = [];
        rng(1);
        
        XTrain = Traindata{:,1};
        YTrain = Traindata{:,2};
        XTest = Testdata{:,1};
        YTest = Testdata{:,2};

        numFilters = optVars.numFilters;
        filterSize = optVars.filterSize;
        numBlocks = optVars.numBlocks;
       
        miniBatchSize = optVars.miniBatchSize;
        learningrate = optVars.learningrate;
        poolSize = optVars.poolSize;

        layers = constructCNN(size(XTrain{1}, 1), numBlocks, filterSize, numFilters,poolSize, 3);

        options = trainingOptions('adam', ...
            'MaxEpochs',Maxepochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'InitialLearnRate',learningrate, ...
            'LearnRateDropPeriod',10, ...
            'LearnRateDropFactor',0.75, ...
            'LearnRateSchedule','piecewise', ...
            'Plots','none',...
            'GradientThreshold',1, ...
            'shuffle','every-epoch',...
            'Verbose',0,...
            'DispatchInBackground',true);
        
        
        net = trainNetwork(XTrain,YTrain,layers,options);
        

        YPred = classify(net, XTest, "MiniBatchSize",miniBatchSize);
        
        %%%%%%%%%%%%%%%%%
        valError = 0;
        for k = 1:length(YPred)
            % fix this
            valError = valError + mean(YPred{k} == YTest{k});
        end
        % return a numeric scalar
        valError = 1 - valError/length(YPred);
        %%%%%%%%%%%%%%%%%%


    end

end
