function lgraph = constructCNN(numFeatures,numBlocks,filterSize,numFilters,poolSize,numResponse)
    arguments
        numFeatures = 1,
        numBlocks = 1,
        filterSize = 5,
        numFilters = 32,
        poolSize = 7,
        numResponse = 3

    end
    
    layer  = [sequenceInputLayer(numFeatures, 'Normalization','zscore','MinLength',512,'Name','sequenceInputLayer')];

    lgraph = layerGraph(layer);
    
    outputName = lgraph.Layers(end).Name;
    
    for i = 1:numBlocks

        layers = [convolution1dLayer(filterSize,numFilters,'PaddingValue','replicate','Padding','same','Name',"conv_"+i)];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"conv_"+i);

        outputName = "conv_"+i;
    end

    layers = [maxPooling1dLayer(poolSize,'Padding','same','Name','pooling')
            fullyConnectedLayer(numResponse,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,'pooling');

    