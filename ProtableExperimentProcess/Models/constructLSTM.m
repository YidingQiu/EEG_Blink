function lgraph = constructLSTM(numFeatures,numBlocks,numHiddenUnits,numResponse,imgInput)
    arguments
        numFeatures = 1,
        numBlocks = 1,
        numHiddenUnits = 32,
        numResponse = 3
        imgInput = false

    end
    
    layer  = [sequenceInputLayer(numFeatures,'Name','sequenceInputLayer')];

    if imgInput

        layer  = [sequenceInputLayer(numFeatures, 'Normalization','zerocenter','Name','sequenceInputLayer')
%                 convolution2dLayer([3 1],numFilter,'Stride',[1 1],'Padding',[1 1 0 0])
                  flattenLayer("Name",'flatten')
                 ]; % fc
    end

    lgraph = layerGraph(layer);
    
    outputName = lgraph.Layers(end).Name;
    
    for i = 1:numBlocks

        layers = [lstmLayer(numHiddenUnits,'OutputMode','sequence','Name',"lstm_"+i)
                layerNormalizationLayer('Name',"layernorm_"+i)];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"lstm_"+i);

        outputName = "layernorm_"+i;
    end

    layers = [fullyConnectedLayer(numResponse,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,'fc');

    