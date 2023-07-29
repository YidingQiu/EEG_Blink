function lgraph = constructLSTMCNN(numFeatures,numLSTMBlocks,numHiddenUnits,numCNNBlocks,filterSize,numFilters,numResponse,imgInput)
    arguments
        numFeatures = 1,
        numLSTMBlocks = 1,
        numHiddenUnits = 32,
        numCNNBlocks = 1,
        filterSize = 3,
        numFilters = 4,
        numResponse = 3,
        imgInput = false

    end
    
    layer  = [sequenceInputLayer(numFeatures, 'Normalization','zerocenter','Name','sequenceInputLayer')];

    if imgInput

        layer  = [sequenceInputLayer(numFeatures, 'Normalization','zerocenter','Name','sequenceInputLayer')
                flattenLayer("Name",'flatten')
                ]; 
    end

    lgraph = layerGraph(layer);
    
    outputName = lgraph.Layers(end).Name;
    
    for i = 1:numLSTMBlocks

        layers = [lstmLayer(numHiddenUnits,'OutputMode','sequence','Name',"lstm_"+i)
                layerNormalizationLayer('Name',"layernorm_"+i)];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"lstm_"+i);

        outputName = "layernorm_"+i;
    end

    for i = 1:numCNNBlocks
        layer = convolution1dLayer(filterSize,numFilters,'Padding','same','Name',"cov_"+i);
        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"cov_"+i);
        outputName = "cov_"+i;
    end


    layers = [fullyConnectedLayer(numResponse,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,'fc');

    