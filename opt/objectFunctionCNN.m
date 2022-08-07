function ObjFcn = objectFunctionCNN(Traindata, Testdata,  Maxepochs)
ObjFcn = @valErrorFun;
    function [valError,emp, net, YPred, TY] = valErrorFun(optVars,mviz)
        emp = [];
        rng(1);
        
        TY = Testdata{1};
        imgsTest = Testdata{2};
        imSize=[221 512];
        numBlocks = optVars.numBlocks;
        filterSizeC= floor((optVars.filterSizeC)/2)*2+1;
        filterSizeS= floor((optVars.filterSizeS)/2)*2+1;
        filterNum= optVars.filterNum;
        dropoutLayerRegularization = optVars.dropoutLayerRegularization;
        learningrate = optVars.learningrate;


        % imSize,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization
        layers = simpleCNNpixel(imSize,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization);

        options = trainingOptions('adam', ...
            'MaxEpochs',Maxepochs, ...
            'MiniBatchSize',64, ...
            'InitialLearnRate',learningrate, ...
            'LearnRateDropPeriod',5, ...
            'LearnRateDropFactor',0.5, ...
            'LearnRateSchedule','piecewise', ...
            'GradientThreshold',1, ...
            'shuffle','every-epoch',...
            'Verbose',0,...
            'DispatchInBackground',true);
        
        
        net = trainNetwork(Traindata,layers,options);
        
        YPred=semanticseg(imgsTest,net);



        PY = [];
        for i = 1:size(YPred,2)      
            PY = [PY;reshape(YPred(:,i),[1 512])];
        end

        
%         %%%%%%%%%%%%%%%%%
%         valError = 0;
%         for k = 1:size(PY,1)
%             % fix this
%             valError = valError + mean(PY{k} == TY{k});
%         end
%         % return a numeric scalar

%         valError = 1 - valError/length(YPred);
        %%%%%%%%%%%%%%%%%%
        %valError = (1-accuracy_eval(YPred, YTest));
        valError = 1-mean(PY == TY,'all');
    end

end
