function lgraph = constructTCN(numFeatures, numFilters, filterSize, numBlocks, dropoutFactor)
    arguments
        numFeatures     = 1
        numFilters      = 16
        filterSize      = 21 
        numBlocks       = 2
        dropoutFactor   = 0.1
    end

    layer  = sequenceInputLayer(numFeatures, Normalization="zscore", Name="sequenceInputLayer");%Normalization="zscore"
    lgraph = layerGraph(layer);
    
    outputName = layer.Name;
    
    for i = 1:numBlocks
        dilationFactor = 2^(i-1);
        
        layers = [
            convolution1dLayer(filterSize, numFilters, DilationFactor=dilationFactor, Padding="causal", Name="conv1_"+i)
            layerNormalizationLayer
            spatialDropoutLayer(dropoutFactor)
            convolution1dLayer(filterSize, numFilters, DilationFactor=dilationFactor, Padding="causal")
            layerNormalizationLayer
            %reluLayer
            swishLayer
            spatialDropoutLayer(dropoutFactor)
            additionLayer(2,Name="add_"+i)];
    
        % Add and connect layers.
        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"conv1_"+i);
    
        % Skip connection.
        if i == 1
            % Include convolution in first skip connection.
            layer = convolution1dLayer(1,numFilters,Name="convSkip");
    
            lgraph = addLayers(lgraph,layer);
            lgraph = connectLayers(lgraph,outputName,"convSkip");
            lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
        else
            lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
        end
        
        % Update layer output name.
        outputName = "add_" + i;
    end
    
    layers = [
        fullyConnectedLayer(4, Name="fc")
        softmaxLayer
        classificationLayer("Name",'classificationLayer')
        % regressionLayer
        ];
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph, outputName, "fc");

end