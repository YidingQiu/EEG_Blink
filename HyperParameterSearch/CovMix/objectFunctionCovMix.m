function ObjFcn = objectFunctionCovMix(Traindata, Testdata,  Maxepochs)
ObjFcn = @valErrorFun;
    function [valError,emp, net, YPred, YTest] = valErrorFun(optVars,~)
        emp = [];
        rng(1);
        
        XTrain = Traindata{:,1};
        YTrain = Traindata{:,2};
        XTest = Testdata{:,1};
        YTest = Testdata{:,2};
    
        try
            miniBatchSize = optVars.miniBatchSize;
            learningrate = optVars.learningrate;
            layers = cwtCovMix(numFeatures = size(XTrain{1},1), ...
                Depth = optVars.Depth, ...
                HiddenDimension = optVars.HiddenDimension, ...
                filterSizeS = optVars.filterSizeS, ...
                response = numel(categories(YTrain{1})), ...
                dropoutLayerRegularization = optVars.dropoutLayerRegularization);
    
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
                'OutputFcn', @outputFunction, ...
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
        catch EM
            valError = 1;
            net = struct;
            YPred = cell(1);
            disp(EM)
        end


    end

end


%% 5 iter 
function stop = outputFunction(info)

persistent bestValidationLoss;
persistent valLag;

if info.State == "start"
    bestValidationLoss = info.ValidationLoss;
    valLag = 0;
elseif ~isempty(info.ValidationLoss)
    if info.ValidationLoss < bestValidationLoss
        valLag = 0;
        bestValidationLoss = info.ValidationLoss;
    else
        valLag = valLag + 1;
    end
    if valLag >= 5
        stop = true;
        return;
    end
end

stop = false;

end
