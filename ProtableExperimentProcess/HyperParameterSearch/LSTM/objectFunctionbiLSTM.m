function ObjFcn = objectFunctionbiLSTM(Traindata, Testdata,  Maxepochs)
ObjFcn = @valErrorFun;
    function [valError,emp, net, YPred, YTest] = valErrorFun(optVars,~)
        emp = [];
        rng(1);
        
        XTrain = Traindata{:,1};
        YTrain = Traindata{:,2};
        XTest = Testdata{:,1};
        YTest = Testdata{:,2};

        miniBatchSize = optVars.miniBatchSize;
        learningrate = optVars.learningrate;
        numFeatures = size(XTrain{1});
        numBlocks = optVars.numBlocks;
        numHiddenUnits = optVars.numHiddenUnits;
        numResponse = numel(categories(YTest{1}));

        try
            if numel(numFeatures) >=3
                numFeatures = [size(XTrain{1},1) size(XTrain{1},2)];
    
                layers = constructbiLSTM(numFeatures,numBlocks,numHiddenUnits,numResponse,true);        
            else
                numFeatures = size(XTrain{1},1);
                layers = constructbiLSTM(numFeatures,numBlocks,numHiddenUnits,numResponse,false); 
            end
    
            
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
        catch
            valError = 1;
            net = [];
            YPred = {};
        end

    end

end
