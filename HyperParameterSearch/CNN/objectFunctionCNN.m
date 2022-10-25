function ObjFcn = objectFunctionCNN(Traindata, Testdata,  Maxepochs)
ObjFcn = @valErrorFun;
    function [valError,emp, net, YPred, TY] = valErrorFun(optVars,~)
        emp = [];
        rng(1);
        
        TY = Testdata{1};
        imgsTest = Testdata{2};
        imSize=size(imread('DataSet\HyperparameterSearch\Train\Img\WTImg\0001_wt.jpg'));
        numBlocks = optVars.numBlocks;
        filterSizeC= floor((optVars.filterSizeC)/2)*2+1;
        filterSizeS= floor((optVars.filterSizeS)/2)*2+1;
        numFilter= optVars.filterNum;
        dropoutLayerRegularization = optVars.dropoutLayerRegularization;
        learningrate = optVars.learningrate;
        activation=optVars.activation;
        batchSize=optVars.batchSize;

        classes = ["blink","noBlink", "muscleArtifact"];
        % imSize,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization,classes,activation
        layers = simpleCNNpixel(imSize,numBlocks,filterSizeC,filterSizeS, ...
            numFilter, dropoutLayerRegularization,classes,activation);%,
        


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
        
        
        net = trainNetwork(Traindata,layers,options);
        
        [YPred,scores,allScores]=semanticseg(imgsTest,net);



        valError = 1-mean(abs(allScores - TY),'all');
%         disp(valError);
    end

end
