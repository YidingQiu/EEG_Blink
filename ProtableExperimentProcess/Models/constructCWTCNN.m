function lgraph = constructCWTCNN(numFeatures,numBlocks,filterSizeC,filterSizeS, filterNum, dropoutLayerRegularization,classes,activation)
    arguments
    numFeatures,
    numBlocks = 3,
    filterSizeC=5,
    filterSizeS =5,
    filterNum=16, 
    dropoutLayerRegularization=0.05,
    classes = ["blink","n/a", "muscle-artifact","no-blink"],
    activation = 'relu'
    end

    switch activation
        case 'tanh'
            activationFunction = tanhLayer();
        case 'swish'
            activationFunction = swishLayer();
        otherwise
            activationFunction = reluLayer();
    end
    

    layers = [
        sequenceInputLayer(numFeatures,'Name','input',"MinLength",512)
        cwtLayer("Name","cwt","SignalLength",512,"VoicesPerOctave",48,"FrequencyLimits",[0.0026041 0.3125])];
    lgraph = layerGraph(layers);
    outputName = "cwt";

    numFeatures = 269; % get this
    paddingSizeS = floor(filterSizeS/2);
    paddingSizeC = floor(filterSizeC/2);
    pool = floor(nthroot(numFeatures,numBlocks))+2;    
    response = 4;%numel(classes);
    

    remains = numFeatures; % size of first S channel
    
    for i = 1:numBlocks
        
        % get size of pooling
        if remains <= pool || i == numBlocks
            pool = remains;
        end
        remains = floor(numFeatures/pool)+1;
        paddingSizeP = floor(pool/2);
        % block
        layers = [               
            convolution2dLayer([filterSizeC filterSizeS], filterNum, ...
            'Padding',[paddingSizeC paddingSizeC paddingSizeS paddingSizeS],'Name',"cov_"+i)

            batchNormalizationLayer
            activationFunction

            maxPooling2dLayer([pool 1],'Stride',[pool 1], ...
            'Padding',[paddingSizeP paddingSizeP 0 0],'Name',"pool_"+i)   
        %     averagePooling2dLayer
            ];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"cov_"+i);
        
        outputName = "pool_"+i;
        
    end  
    
    layers = [ 
        convolution2dLayer([1 filterSizeS],response,'Padding',[0 0 paddingSizeS paddingSizeS],"Name",'fullCov')
        dropoutLayer(dropoutLayerRegularization,'Name','Dropout')
        softmaxLayer('Name','softMax')
        classificationLayer("Classes",'auto')];                
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,'fullCov');



end
