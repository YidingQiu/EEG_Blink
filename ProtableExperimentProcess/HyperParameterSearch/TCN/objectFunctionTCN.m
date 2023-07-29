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
        dropoutFactor = optVars.dropoutFactor;
        miniBatchSize = optVars.miniBatchSize;
        learningrate = optVars.learningrate;

        layers = constructTCN(size(XTrain{1}, 1), numFilters, filterSize, numBlocks, dropoutFactor);

        options = trainingOptions('adam', ...
            'MaxEpochs',Maxepochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'InitialLearnRate',learningrate, ...
            'LearnRateDropPeriod',5, ...
            'LearnRateDropFactor',0.5, ...
            'LearnRateSchedule','piecewise', ...
            'Plots','none',...
            'GradientThreshold',1, ...
            'shuffle','every-epoch',...
            'Verbose',0,...
            'DispatchInBackground',true);
        
        try
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
        catch
            valError = 1;
            net = [];
            YPred = {};
        end


    end

end
