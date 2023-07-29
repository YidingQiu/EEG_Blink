function ObjFcn = objectFunctionCNN(Traindata, Testdata,  Maxepochs,i)
ObjFcn = @valErrorFun;
    function [valError,emp, net, YPred, YTest] = valErrorFun(optVars,~)
        emp = [];
        rng(1);
        
        XTrain = Traindata{:,1};
        YTrain = Traindata{:,2};
        XTest = Testdata{:,1};
        YTest = Testdata{:,2};

        numBlocks = optVars.numBlocks;
        filterSizeC= floor((optVars.filterSizeC)/2)*2+1;
        filterSizeS= floor((optVars.filterSizeS)/2)*2+1;
        numFilter= optVars.filterNum;
        dropoutLayerRegularization = optVars.dropoutLayerRegularization;
        learningrate = optVars.learningrate;
        batchSize=optVars.batchSize;
       
        % try
            % imSize,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization,classes,activation
            classes = ['closing', 'opening', 'n/a', 'muscle-artifact'];
            %    numFeatures,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization,classes,activation
            layers = constructCWTCNN(i,numBlocks,filterSizeC,filterSizeS, numFilter, dropoutLayerRegularization,classes,"relu");
    
    
            options = trainingOptions('adam', ...
                'MaxEpochs',Maxepochs, ...
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
            
            net = trainNetwork(XTrain,YTrain,layers,options);
            
            YPred = classify(net, XTest, "MiniBatchSize",miniBatchSize);
            
            valError = 0;
            for k = 1:length(YPred)
                % fix this
                valError = valError + mean(YPred{k} == YTest{k});
            end
            % return a numeric scalar
            valError = 1 - valError/length(YPred);

        % catch EM
        %     disp(EM)
        %     valError = 1;
        %     net = [];
        %     YPred = {};
        % end
    end

end
