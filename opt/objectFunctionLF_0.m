function ObjFcn = objectFunctionLF_0(Traindata, Testdata,  Maxepochs)
ObjFcn = @valErrorFun;
    function [valError,emp, net, Ys, YPred, YPredStep, YTest] = valErrorFun(optVars,mviz)
        emp = [];
        
        XTrain = Traindata{:,1};
        YTrain = Traindata{:,2};
        XTest = Testdata{:,1};
        YTest = Testdata{:,2};
        numResponses = 4;
        numHiddenUnits = optVars.HiddenUnits;
%         dropOutRate = str2double(optVars.DropOutRate);
        miniBatchSize = 8;
%         miniBatchSize = str2double(char(optVars.MiniBatchSize));

        
        layers = [ ...            
            sequenceInputLayer(1)
            lstmLayer(numHiddenUnits,'OutputMode','sequence')
            fullyConnectedLayer(numResponses)
            softmaxLayer
            classificationLayer
            ];
        
        options = trainingOptions('adam', ...
            'MaxEpochs',Maxepochs, ...
            'MiniBatchSize',miniBatchSize, ...
            'InitialLearnRate',optVars.learningrate, ...
            'LearnRateDropPeriod',10, ...
            'LearnRateDropFactor',0.5, ...
            'LearnRateSchedule','piecewise', ...
            'GradientThreshold',1, ...
            'Shuffle','every-epoch', ...%             'Plots','training-progress',...
            'ExecutionEnvironment','gpu', ...
            'Verbose',0);
            
        
        net = trainNetwork(XTrain,YTrain,layers,options);
        

        YPred = classify(net, XTest, "MiniBatchSize",miniBatchSize);

        valError = 0;
        for k = 1:length(YPred)
            % fix this
            valError = valError + mean(YPred{k} == YTest{k})/length(YTest{k});
        end
        % return a numeric scalar
        
        valError = 1-valError/length(YPred);
%         disp(valError);
    end

end
